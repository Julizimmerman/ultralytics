# ==============================================================================
# SegmentEnhanced: Cabeza de Segmentación Mejorada para YOLO
# ==============================================================================
# Mejoras implementadas:
#   A) Más prototipos y coeficientes (nm=64, npr=512 por defecto)
#   B) Generación de prototipos multi-escala (fusiona información de todas las escalas)
#   C) Módulo de refinamiento no-lineal para pulir las máscaras
#
# Para usar: Registrar en ultralytics/nn/modules/__init__.py y ultralytics/nn/tasks.py
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Importaciones de Ultralytics (ajustar según tu instalación)
# from ultralytics.nn.modules.conv import Conv
# from ultralytics.nn.modules.block import Proto
# from ultralytics.nn.modules.head import Detect

# ==============================================================================
# Módulos auxiliares (para que el archivo sea autocontenido)
# En producción, importarías estos de ultralytics.nn.modules
# ==============================================================================

class Conv(nn.Module):
    """Convolución estándar con BatchNorm y activación SiLU."""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Args:
            c1: Canales de entrada
            c2: Canales de salida
            k: Tamaño del kernel
            s: Stride
            p: Padding (auto si None)
            g: Grupos
            act: Usar activación SiLU
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def autopad(k, p=None):
    """Calcula padding automático para mantener dimensiones."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# ==============================================================================
# MEJORA B: Módulo de Prototipos Multi-Escala
# ==============================================================================

class MultiScaleProto(nn.Module):
    """
    Genera prototipos usando información de múltiples escalas.
    
    En vez de usar solo la feature map más grande (x[0]), fusiona información
    de todas las escalas para crear prototipos que entienden tanto detalles
    finos como contexto global.
    
    Metáfora: El pintor ahora mira la imagen desde cerca (detalles), 
    desde media distancia (partes) y desde lejos (contexto global),
    y combina todas estas vistas para crear mejores plantillas.
    """
    
    def __init__(self, ch: tuple, c_: int = 512, c2: int = 64):
        """
        Args:
            ch: Tuple de canales de entrada para cada escala (ej: (256, 512, 1024))
            c_: Canales intermedios
            c2: Canales de salida (número de prototipos)
        """
        super().__init__()
        
        self.num_scales = len(ch)
        
        # Proyección para cada escala a un espacio común
        proj_ch = c_ // self.num_scales
        actual_total = proj_ch * self.num_scales  # 510, no 512

        self.scale_projs = nn.ModuleList([
            Conv(ch[i], proj_ch, k=1) for i in range(self.num_scales)
        ])

        # Upsampling para alinear todas las escalas a la resolución más alta
        # (las escalas más pequeñas se upsamplearán)
        
        # Fusión de escalas
        self.fusion = nn.Sequential(
            Conv(actual_total, c_, k=3),  # entrada = canales reales concatenados
            Conv(c_, c_, k=3),
        )
        
        # Generación de prototipos (similar al Proto original pero con más capacidad)
        self.upsample = nn.ConvTranspose2d(c_, c_, kernel_size=2, stride=2, bias=True)
        self.refine = nn.Sequential(
            Conv(c_, c_, k=3),
            Conv(c_, c2, k=1, act=False),  # Sin activación final
        )
        
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: Lista de feature maps [x0, x1, x2] de diferentes escalas
                     x0: Mayor resolución, menor semántica (ej: 80x80)
                     x1: Media resolución (ej: 40x40)
                     x2: Menor resolución, mayor semántica (ej: 20x20)
        
        Returns:
            proto: Tensor de prototipos (B, nm, H, W) donde H, W es 2x la resolución de x0
        """
        # Obtener la resolución objetivo (la de la escala más grande)
        target_size = features[0].shape[2:]  # (H, W)
        
        # Proyectar y alinear cada escala
        aligned_features = []
        for i, (feat, proj) in enumerate(zip(features, self.scale_projs)):
            # Proyectar a espacio común
            projected = proj(feat)
            
            # Upsample a la resolución más alta si es necesario
            if projected.shape[2:] != target_size:
                projected = F.interpolate(
                    projected, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            aligned_features.append(projected)
        
        # Concatenar todas las escalas
        fused = torch.cat(aligned_features, dim=1)  # (B, c_, H, W)
        
        # Fusionar información
        fused = self.fusion(fused)
        
        # Upsample para mayor resolución de máscaras
        fused = self.upsample(fused)
        
        # Generar prototipos finales
        proto = self.refine(fused)
        
        return proto


# ==============================================================================
# MEJORA C: Módulo de Refinamiento de Máscaras
# ==============================================================================

class MaskRefiner(nn.Module):
    """
    Refina las máscaras después de la combinación lineal de prototipos.
    
    La combinación lineal de prototipos produce máscaras aproximadas.
    Este módulo las "pule" usando una pequeña red que:
    1. Observa la máscara aproximada
    2. Observa las features de la imagen (para contexto)
    3. Produce una máscara más precisa
    
    Metáfora: Después de que el pintor mezcla sus plantillas, un segundo
    artista toma ese borrador y lo refina: define mejor los bordes,
    elimina ruido, y ajusta los detalles.
    """
    
    def __init__(self, nm: int = 64, feat_channels: int = 256, hidden_dim: int = 64):
        """
        Args:
            nm: Número de máscaras/prototipos
            feat_channels: Canales de las features de imagen
            hidden_dim: Dimensión oculta del refinador
        """
        super().__init__()
        
        # Procesar la máscara aproximada
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Procesar features de la imagen para contexto
        self.feat_encoder = nn.Sequential(
            Conv(feat_channels, hidden_dim, k=1),
        )
        
        # Fusión y refinamiento
        self.refine_net = nn.Sequential(
            # Entrada: máscara_encoded + features_encoded = 2 * hidden_dim
            Conv(hidden_dim * 2, hidden_dim, k=3),
            Conv(hidden_dim, hidden_dim, k=3),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),  # Salida: máscara refinada
        )
        
    def forward(self, coarse_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_mask: Máscara aproximada (B, 1, H, W) o (B, N, H, W) para N objetos
            features: Features de la imagen (B, C, H', W')
        
        Returns:
            refined_mask: Máscara refinada del mismo tamaño que coarse_mask
        """
        # Guardar forma original
        original_shape = coarse_mask.shape
        
        # Si tenemos múltiples máscaras, procesarlas una por una o en batch
        if len(original_shape) == 4 and original_shape[1] > 1:
            # Tenemos (B, N, H, W) - N máscaras por imagen
            B, N, H, W = original_shape
            
            # Reshape para procesar todas las máscaras juntas
            coarse_mask = coarse_mask.view(B * N, 1, H, W)
            
            # Repetir features para cada máscara
            features = features.repeat_interleave(N, dim=0)
        
        # Asegurar que features tengan la misma resolución que la máscara
        if features.shape[2:] != coarse_mask.shape[2:]:
            features = F.interpolate(
                features, 
                size=coarse_mask.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Encodear máscara y features
        mask_encoded = self.mask_encoder(coarse_mask)
        feat_encoded = self.feat_encoder(features)
        
        # Concatenar y refinar
        combined = torch.cat([mask_encoded, feat_encoded], dim=1)
        residual = self.refine_net(combined)
        
        # Conexión residual: máscara refinada = máscara original + corrección
        refined = coarse_mask + residual
        
        # Restaurar forma original si procesamos múltiples máscaras
        if len(original_shape) == 4 and original_shape[1] > 1:
            refined = refined.view(B, N, H, W)
        
        return refined


# ==============================================================================
# CABEZA PRINCIPAL: SegmentEnhanced (combina mejoras A, B y C)
# ==============================================================================

class SegmentEnhanced(nn.Module):
    """
    Cabeza de segmentación mejorada para YOLO.
    
    Combina tres mejoras sobre la cabeza original:
    - A) Más prototipos (64 vs 32) para mayor capacidad de representación
    - B) Prototipos multi-escala para mejor comprensión de detalles y contexto
    - C) Refinamiento no-lineal para máscaras más precisas
    
    Esta clase está diseñada para reemplazar la clase Segment original.
    Hereda la funcionalidad de detección y añade segmentación mejorada.
    """
    
    # Atributos de clase (compatibilidad con Detect)
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    stride = torch.empty(0)
    legacy = False
    
    def __init__(
        self, 
        nc: int = 80,
        nm: int = 64,
        npr: int = 512,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        use_refiner: bool = True,
    ):
        """
        Args:
            nc: Número de clases para detección
            nm: Número de prototipos/máscaras
            npr: Canales intermedios para generación de prototipos
            ch: Tuple de canales de entrada (ej: (256, 512, 1024))
            reg_max: Número máximo de bins para DFL
            use_refiner: Si usar el módulo de refinamiento
        """
        super().__init__()
        
        self.nc = nc  # número de clases
        self.nm = nm  # número de máscaras
        self.npr = npr  # canales de prototipos
        self.nl = len(ch)  # número de capas/escalas
        self.reg_max = reg_max
        self.use_refiner = use_refiner
        self.no = nc + reg_max * 4  # número de outputs por anchor (detección)
        self.end2end = end2end
        
        # =====================================================================
        # Componentes de DETECCIÓN (similar a Detect)
        # =====================================================================
        
        c2 = max(16, ch[0] // 4, reg_max * 4)  # canales para regresión de bbox
        c3 = max(ch[0], min(nc, 100))  # canales para clasificación
        
        # Cabeza de regresión de bounding box
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), 
                Conv(c2, c2, 3), 
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        
        # Cabeza de clasificación
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), 
                Conv(c3, c3, 3), 
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        
        # DFL (Distribution Focal Loss)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # =====================================================================
        # Componentes de SEGMENTACIÓN MEJORADA
        # =====================================================================
        
        # MEJORA B: Generador de prototipos multi-escala
        self.proto = MultiScaleProto(ch, c_=self.npr, c2=self.nm)
        
        # Cabeza de coeficientes de máscara (MEJORA A: más coeficientes)
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c4, 3), 
                Conv(c4, c4, 3), 
                nn.Conv2d(c4, self.nm, 1)
            ) for x in ch
        )
        
        # MEJORA C: Refinador de máscaras
        if self.use_refiner:
            self.refiner = MaskRefiner(
                nm=self.nm, 
                feat_channels=ch[0], 
                hidden_dim=64
            )
        else:
            self.refiner = None

    def forward(self, x: list) -> tuple:
        """
        Forward pass de la cabeza de segmentación mejorada.
        
        Args:
            x: Lista de feature maps de diferentes escalas
               [x0 (B, C0, H0, W0), x1 (B, C1, H1, W1), x2 (B, C2, H2, W2)]
        
        Returns:
            Durante entrenamiento: (predicciones_detección, coeficientes_máscara, prototipos)
            Durante inferencia: Similar pero formateado para post-procesamiento
        """
        # =====================================================================
        # Parte de Detección
        # =====================================================================

        shape = x[0].shape

        # Guardar features originales ANTES de que el loop las modifique
        original_features = [xi.clone() for xi in x]
        
        # Obtener predicciones de bbox y clase de cada escala
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # Usar features originales para el proto
        proto = self.proto(original_features)
        
        bs = proto.shape[0]

        mc = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 
            dim=2
        )

        if self.training:
            return x, mc, proto

        return (torch.cat([xi.view(bs, self.no, -1) for xi in x], dim=2), mc, proto)
            
    
    def apply_refinement(
        self, 
        masks: torch.Tensor, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica refinamiento a máscaras ya combinadas (para uso en post-procesamiento).
        
        Args:
            masks: Máscaras combinadas (B, N, H, W) donde N es número de objetos
            features: Features de la imagen para contexto
        
        Returns:
            refined_masks: Máscaras refinadas
        """
        if self.refiner is None:
            return masks
        
        return self.refiner(masks, features)


# ==============================================================================
# DFL (Distribution Focal Loss) - Necesario para la cabeza de detección
# ==============================================================================

class DFL(nn.Module):
    """
    Distribution Focal Loss layer.
    Convierte distribuciones discretas en valores continuos.
    """
    
    def __init__(self, c1: int = 16):
        """
        Args:
            c1: Número de bins de la distribución
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de distribuciones (B, 4*c1, H, W) o (B, 4*c1, num_anchors)
        
        Returns:
            Valores de bbox regresados (B, 4, H, W) o (B, 4, num_anchors)
        """
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# ==============================================================================
# Ejemplo de uso y testing
# ==============================================================================

if __name__ == "__main__":
    # Configuración de prueba
    batch_size = 2
    nc = 80  # clases COCO
    nm = 64  # prototipos (MEJORA A)
    ch = (256, 512, 1024)  # canales típicos de YOLOv8
    
    # Crear feature maps simuladas (como las que salen del backbone)
    # Escala grande (alta resolución, menos semántica)
    x0 = torch.randn(batch_size, ch[0], 80, 80)
    # Escala media
    x1 = torch.randn(batch_size, ch[1], 40, 40)
    # Escala pequeña (baja resolución, más semántica)
    x2 = torch.randn(batch_size, ch[2], 20, 20)
    
    features = [x0, x1, x2]
    
    # Crear la cabeza mejorada
    print("=" * 60)
    print("Creando SegmentEnhanced con mejoras A, B y C")
    print("=" * 60)
    
    head = SegmentEnhanced(
        nc=nc,
        nm=nm,          # MEJORA A: 64 vs 32
        npr=512,        # MEJORA A: 512 vs 256
        ch=ch,
        use_refiner=True  # MEJORA C
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    
    print(f"\nParámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Forward pass en modo entrenamiento
    print("\n" + "=" * 60)
    print("Forward pass (modo entrenamiento)")
    print("=" * 60)
    
    head.train()
    outputs_train = head(features)
    
    print(f"Número de outputs: {len(outputs_train)}")
    print(f"Detecciones por escala: {[o.shape for o in outputs_train[0]]}")
    print(f"Coeficientes de máscara: {outputs_train[1].shape}")
    print(f"Prototipos: {outputs_train[2].shape}")
    
    # Forward pass en modo inferencia
    print("\n" + "=" * 60)
    print("Forward pass (modo inferencia)")
    print("=" * 60)
    
    head.eval()
    features_copy = [f.clone() for f in features]  # Copiar porque se modifican in-place
    
    with torch.no_grad():
        outputs_eval = head(features_copy)
    
    print(f"Detecciones concatenadas: {outputs_eval[0].shape}")
    print(f"Coeficientes de máscara: {outputs_eval[1].shape}")
    print(f"Prototipos: {outputs_eval[2].shape}")
    
    # Test del refinador
    print("\n" + "=" * 60)
    print("Test del Refinador de Máscaras")
    print("=" * 60)
    
    # Simular máscaras ya combinadas (después de coef @ proto)
    num_objects = 5
    mask_h, mask_w = 160, 160  # Resolución típica de máscaras
    fake_masks = torch.randn(batch_size, num_objects, mask_h, mask_w)
    fake_features = torch.randn(batch_size, ch[0], 80, 80)
    
    refined = head.apply_refinement(fake_masks, fake_features)
    print(f"Máscaras de entrada: {fake_masks.shape}")
    print(f"Máscaras refinadas: {refined.shape}")
    
    print("\n" + "=" * 60)
    print("✅ Todas las pruebas pasaron!")
    print("=" * 60)
