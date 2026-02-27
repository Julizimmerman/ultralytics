# # ==============================================================================
# # SegmentEnhanced: Cabeza de Segmentación Mejorada para YOLO
# # ==============================================================================
# # Mejoras implementadas:
# #   A) Más prototipos y coeficientes (nm=64, npr=512 por defecto)
# #   B) Generación de prototipos multi-escala (fusiona información de todas las escalas)
# #   C) Módulo de refinamiento no-lineal para pulir las máscaras
# #
# # Para usar: Registrar en ultralytics/nn/modules/__init__.py y ultralytics/nn/tasks.py
# # ==============================================================================

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import copy

# class Conv(nn.Module):
#     """Convolución estándar con BatchNorm y activación SiLU."""
    
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         """
#         Args:
#             c1: Canales de entrada
#             c2: Canales de salida
#             k: Tamaño del kernel
#             s: Stride
#             p: Padding (auto si None)
#             g: Grupos
#             act: Usar activación SiLU
#         """
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))


# def autopad(k, p=None):
#     """Calcula padding automático para mantener dimensiones."""
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
#     return p


# # ==============================================================================
# # MEJORA: Módulo de Prototipos Multi-Escala
# # ==============================================================================

# class MultiScaleProto(nn.Module):
#     """
#     Genera prototipos usando información de múltiples escalas.
    
#     En vez de usar solo la feature map más grande (x[0]), fusiona información
#     de todas las escalas para crear prototipos que entienden tanto detalles
#     finos como contexto global.
    
#     Metáfora: El pintor ahora mira la imagen desde cerca (detalles), 
#     desde media distancia (partes) y desde lejos (contexto global),
#     y combina todas estas vistas para crear mejores plantillas.
#     """
    
#     def __init__(self, ch: tuple, c_: int = 512, c2: int = 64):
#         """
#         Args:
#             ch: Tuple de canales de entrada para cada escala (ej: (256, 512, 1024))
#             c_: Canales intermedios
#             c2: Canales de salida (número de prototipos)
#         """
#         super().__init__()
        
#         self.num_scales = len(ch)
        
#         # Proyección para cada escala a un espacio común
#         self.scale_projs = nn.ModuleList([
#             Conv(ch[i], c_ // self.num_scales, k=1) for i in range(self.num_scales)
#         ])
        
#         # Upsampling para alinear todas las escalas a la resolución más alta
#         # (las escalas más pequeñas se upsamplearán)
        
#         # Fusión de escalas
#         self.fusion = nn.Sequential(
#             Conv(c_, c_, k=3),
#             Conv(c_, c_, k=3),
#         )
        
#         # Generación de prototipos (similar al Proto original pero con más capacidad)
#         self.upsample = nn.ConvTranspose2d(c_, c_, kernel_size=2, stride=2, bias=True)
#         self.refine = nn.Sequential(
#             Conv(c_, c_, k=3),
#             Conv(c_, c2, k=1, act=False),  # Sin activación final
#         )
        
#     def forward(self, features: list) -> torch.Tensor:
#         """
#         Args:
#             features: Lista de feature maps [x0, x1, x2] de diferentes escalas
#                      x0: Mayor resolución, menor semántica (ej: 80x80)
#                      x1: Media resolución (ej: 40x40)
#                      x2: Menor resolución, mayor semántica (ej: 20x20)
        
#         Returns:
#             proto: Tensor de prototipos (B, nm, H, W) donde H, W es 2x la resolución de x0
#         """
#         # Obtener la resolución objetivo (la de la escala más grande)
#         target_size = features[0].shape[2:]  # (H, W)
        
#         # Proyectar y alinear cada escala
#         aligned_features = []
#         for i, (feat, proj) in enumerate(zip(features, self.scale_projs)):
#             # Proyectar a espacio común
#             projected = proj(feat)
            
#             # Upsample a la resolución más alta si es necesario
#             if projected.shape[2:] != target_size:
#                 projected = F.interpolate(
#                     projected, 
#                     size=target_size, 
#                     mode='bilinear', 
#                     align_corners=False
#                 )
            
#             aligned_features.append(projected)
        
#         # Concatenar todas las escalas
#         fused = torch.cat(aligned_features, dim=1)  # (B, c_, H, W)
        
#         # Fusionar información
#         fused = self.fusion(fused)
        
#         # Upsample para mayor resolución de máscaras
#         fused = self.upsample(fused)
        
#         # Generar prototipos finales
#         proto = self.refine(fused)
        
#         return proto


# # ==============================================================================
# # MEJORA: Módulo de Refinamiento de Máscaras
# # ==============================================================================

# class MaskRefiner(nn.Module):
#     """
#     Refina las máscaras después de la combinación lineal de prototipos.
    
#     La combinación lineal de prototipos produce máscaras aproximadas.
#     Este módulo las "pule" usando una pequeña red que:
#     1. Observa la máscara aproximada
#     2. Observa las features de la imagen (para contexto)
#     3. Produce una máscara más precisa
    
#     Metáfora: Después de que el pintor mezcla sus plantillas, un segundo
#     artista toma ese borrador y lo refina: define mejor los bordes,
#     elimina ruido, y ajusta los detalles.
#     """
    
#     def __init__(self, nm: int = 64, feat_channels: int = 256, hidden_dim: int = 64):
#         """
#         Args:
#             nm: Número de máscaras/prototipos
#             feat_channels: Canales de las features de imagen
#             hidden_dim: Dimensión oculta del refinador
#         """
#         super().__init__()
        
#         # Procesar la máscara aproximada
#         self.mask_encoder = nn.Sequential(
#             nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#         )
        
#         # Procesar features de la imagen para contexto
#         self.feat_encoder = nn.Sequential(
#             Conv(feat_channels, hidden_dim, k=1),
#         )
        
#         # Fusión y refinamiento
#         self.refine_net = nn.Sequential(
#             # Entrada: máscara_encoded + features_encoded = 2 * hidden_dim
#             Conv(hidden_dim * 2, hidden_dim, k=3),
#             Conv(hidden_dim, hidden_dim, k=3),
#             nn.Conv2d(hidden_dim, 1, kernel_size=1),  # Salida: máscara refinada
#         )
        
#     def forward(self, coarse_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             coarse_mask: Máscara aproximada (B, 1, H, W) o (B, N, H, W) para N objetos
#             features: Features de la imagen (B, C, H', W')
        
#         Returns:
#             refined_mask: Máscara refinada del mismo tamaño que coarse_mask
#         """
#         # Guardar forma original
#         original_shape = coarse_mask.shape
        
#         # Si tenemos múltiples máscaras, procesarlas una por una o en batch
#         if len(original_shape) == 4 and original_shape[1] > 1:
#             # Tenemos (B, N, H, W) - N máscaras por imagen
#             B, N, H, W = original_shape
            
#             # Reshape para procesar todas las máscaras juntas
#             coarse_mask = coarse_mask.view(B * N, 1, H, W)
            
#             # Repetir features para cada máscara
#             features = features.repeat_interleave(N, dim=0)
        
#         # Asegurar que features tengan la misma resolución que la máscara
#         if features.shape[2:] != coarse_mask.shape[2:]:
#             features = F.interpolate(
#                 features, 
#                 size=coarse_mask.shape[2:], 
#                 mode='bilinear', 
#                 align_corners=False
#             )
        
#         # Encodear máscara y features
#         mask_encoded = self.mask_encoder(coarse_mask)
#         feat_encoded = self.feat_encoder(features)
        
#         # Concatenar y refinar
#         combined = torch.cat([mask_encoded, feat_encoded], dim=1)
#         residual = self.refine_net(combined)
        
#         # Conexión residual: máscara refinada = máscara original + corrección
#         refined = coarse_mask + residual
        
#         # Restaurar forma original si procesamos múltiples máscaras
#         if len(original_shape) == 4 and original_shape[1] > 1:
#             refined = refined.view(B, N, H, W)
        
#         return refined


# # ==============================================================================
# # CABEZA PRINCIPAL: SegmentEnhanced (combina las tres mejoras)
# # ==============================================================================

# class SegmentEnhanced(nn.Module):
#     """
#     Cabeza de segmentación mejorada para YOLO.
    
#     Combina tres mejoras sobre la cabeza original:
#     - A) Más prototipos (64 vs 32) para mayor capacidad de representación
#     - B) Prototipos multi-escala para mejor comprensión de detalles y contexto
#     - C) Refinamiento no-lineal para máscaras más precisas
    
#     Esta clase está diseñada para reemplazar la clase Segment original.
#     Hereda la funcionalidad de detección y añade segmentación mejorada.
#     """
    
#     # Atributos de clase (compatibilidad con Detect)
#     dynamic = False
#     export = False
#     format = None
#     end2end = False
#     max_det = 300
#     shape = None
#     anchors = torch.empty(0)
#     strides = torch.empty(0)
#     legacy = False
    
#     def __init__(
#         self, 
#         nc: int = 80,
#         nm: int = 64,
#         npr: int = 512,
#         reg_max: int = 16,
#         end2end: bool = False,
#         ch: tuple = (),
#         use_refiner: bool = True,
#     ):
#         """
#         Args:
#             nc: Número de clases para detección
#             nm: Número de prototipos/máscaras
#             npr: Canales intermedios para generación de prototipos
#             ch: Tuple de canales de entrada (ej: (256, 512, 1024))
#             reg_max: Número máximo de bins para DFL
#             use_refiner: Si usar el módulo de refinamiento
#         """
#         super().__init__()
        
#         self.nc = nc  # número de clases
#         self.nm = nm  # número de máscaras
#         self.npr = npr  # canales de prototipos
#         self.nl = len(ch)  # número de capas/escalas
#         self.reg_max = reg_max
#         self.use_refiner = use_refiner
#         self.no = nc + reg_max * 4  # número de outputs por anchor (detección)
#         self.end2end = end2end
        
#         # =====================================================================
#         # Componentes de DETECCIÓN (similar a Detect)
#         # =====================================================================
        
#         c2 = max(16, ch[0] // 4, reg_max * 4)  # canales para regresión de bbox
#         c3 = max(ch[0], min(nc, 100))  # canales para clasificación
        
#         # Cabeza de regresión de bounding box
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(
#                 Conv(x, c2, 3), 
#                 Conv(c2, c2, 3), 
#                 nn.Conv2d(c2, 4 * self.reg_max, 1)
#             ) for x in ch
#         )
        
#         # Cabeza de clasificación
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(
#                 Conv(x, c3, 3), 
#                 Conv(c3, c3, 3), 
#                 nn.Conv2d(c3, self.nc, 1)
#             ) for x in ch
#         )
        
#         # DFL (Distribution Focal Loss)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
#         # =====================================================================
#         # Componentes de SEGMENTACIÓN MEJORADA
#         # =====================================================================
        
#         # MEJORA B: Generador de prototipos multi-escala
#         self.proto = MultiScaleProto(ch, c_=self.npr, c2=self.nm)
        
#         # Cabeza de coeficientes de máscara (MEJORA A: más coeficientes)
#         c4 = max(ch[0] // 4, self.nm)
#         self.cv4 = nn.ModuleList(
#             nn.Sequential(
#                 Conv(x, c4, 3), 
#                 Conv(c4, c4, 3), 
#                 nn.Conv2d(c4, self.nm, 1)
#             ) for x in ch
#         )
        
#         # MEJORA C: Refinador de máscaras
#         if self.use_refiner:
#             self.refiner = MaskRefiner(
#                 nm=self.nm, 
#                 feat_channels=ch[0], 
#                 hidden_dim=64
#             )
#         else:
#             self.refiner = None
            
#     def forward(self, x: list) -> tuple:
#         """
#         Forward pass de la cabeza de segmentación mejorada.
        
#         Args:
#             x: Lista de feature maps de diferentes escalas
#                [x0 (B, C0, H0, W0), x1 (B, C1, H1, W1), x2 (B, C2, H2, W2)]
        
#         Returns:
#             Durante entrenamiento: (predicciones_detección, coeficientes_máscara, prototipos)
#             Durante inferencia: Similar pero formateado para post-procesamiento
#         """
#         # =====================================================================
#         # Parte de Detección
#         # =====================================================================
        
#         shape = x[0].shape  # (B, C, H, W)
        
#         # Obtener predicciones de bbox y clase de cada escala
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
#         # =====================================================================
#         # Parte de Segmentación Mejorada
#         # =====================================================================
        
#         # Guardar features originales para el refinador
#         features_for_refiner = x[0] if self.use_refiner else None
        
#         # MEJORA B: Generar prototipos multi-escala
#         # Necesitamos las features originales, no las modificadas
#         # Por eso guardamos una referencia antes de modificar x
#         proto = self.proto(x)  # (B, nm, H*2, W*2)
        
#         bs = proto.shape[0]  # batch size
        
#         # Obtener coeficientes de máscara de cada escala
#         mc = torch.cat(
#             [self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 
#             dim=2
#         )  # (B, nm, num_anchors_total)
        
#         # =====================================================================
#         # Durante entrenamiento, retornar componentes separados
#         # =====================================================================
        
#         if self.training:
#             return x, mc, proto
        
#         # =====================================================================
#         # Durante inferencia, aplicar refinamiento si está habilitado
#         # =====================================================================
        
#         # Nota: El refinamiento completo se aplica típicamente en post-procesamiento
#         # cuando ya tenemos las máscaras individuales por objeto.
#         # Aquí retornamos los componentes necesarios.
        
#         return (torch.cat([xi.view(bs, self.no, -1) for xi in x], dim=2), mc, proto)
    
#     def apply_refinement(
#         self, 
#         masks: torch.Tensor, 
#         features: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Aplica refinamiento a máscaras ya combinadas (para uso en post-procesamiento).
        
#         Args:
#             masks: Máscaras combinadas (B, N, H, W) donde N es número de objetos
#             features: Features de la imagen para contexto
        
#         Returns:
#             refined_masks: Máscaras refinadas
#         """
#         if self.refiner is None:
#             return masks
        
#         return self.refiner(masks, features)


# # ==============================================================================
# # DFL (Distribution Focal Loss) - Necesario para la cabeza de detección
# # ==============================================================================

# class DFL(nn.Module):
#     """
#     Distribution Focal Loss layer.
#     Convierte distribuciones discretas en valores continuos.
#     """
    
#     def __init__(self, c1: int = 16):
#         """
#         Args:
#             c1: Número de bins de la distribución
#         """
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(c1, dtype=torch.float)
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         self.c1 = c1

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor de distribuciones (B, 4*c1, H, W) o (B, 4*c1, num_anchors)
        
#         Returns:
#             Valores de bbox regresados (B, 4, H, W) o (B, 4, num_anchors)
#         """
#         b, c, a = x.shape  # batch, channels, anchors
#         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# # ==============================================================================
# # Uso y testing
# # ==============================================================================

# if __name__ == "__main__":
#     # Configuración de prueba
#     batch_size = 2
#     nc = 80  # clases COCO
#     nm = 64  # prototipos (MEJORA A)
#     ch = (256, 512, 1024)  # canales típicos de YOLOv8
    
#     # Crear feature maps simuladas (como las que salen del backbone)
#     # Escala grande (alta resolución, menos semántica)
#     x0 = torch.randn(batch_size, ch[0], 80, 80)
#     # Escala media
#     x1 = torch.randn(batch_size, ch[1], 40, 40)
#     # Escala pequeña (baja resolución, más semántica)
#     x2 = torch.randn(batch_size, ch[2], 20, 20)
    
#     features = [x0, x1, x2]
    
#     # Crear la cabeza mejorada
#     print("=" * 60)
#     print("Creando SegmentEnhanced con mejoras A, B y C")
#     print("=" * 60)
    
#     head = SegmentEnhanced(
#         nc=nc,
#         nm=nm,          # MEJORA A: 64 vs 32
#         npr=512,        # MEJORA A: 512 vs 256
#         ch=ch,
#         use_refiner=True  # MEJORA C
#     )
    
#     # Contar parámetros
#     total_params = sum(p.numel() for p in head.parameters())
#     trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    
#     print(f"\nParámetros totales: {total_params:,}")
#     print(f"Parámetros entrenables: {trainable_params:,}")
    
#     # Forward pass en modo entrenamiento
#     print("\n" + "=" * 60)
#     print("Forward pass (modo entrenamiento)")
#     print("=" * 60)
    
#     head.train()
#     outputs_train = head(features)
    
#     print(f"Número de outputs: {len(outputs_train)}")
#     print(f"Detecciones por escala: {[o.shape for o in outputs_train[0]]}")
#     print(f"Coeficientes de máscara: {outputs_train[1].shape}")
#     print(f"Prototipos: {outputs_train[2].shape}")
    
#     # Forward pass en modo inferencia
#     print("\n" + "=" * 60)
#     print("Forward pass (modo inferencia)")
#     print("=" * 60)
    
#     head.eval()
#     features_copy = [f.clone() for f in features]  # Copiar porque se modifican in-place
    
#     with torch.no_grad():
#         outputs_eval = head(features_copy)
    
#     print(f"Detecciones concatenadas: {outputs_eval[0].shape}")
#     print(f"Coeficientes de máscara: {outputs_eval[1].shape}")
#     print(f"Prototipos: {outputs_eval[2].shape}")
    
#     # Test del refinador
#     print("\n" + "=" * 60)
#     print("Test del Refinador de Máscaras")
#     print("=" * 60)
    
#     # Simular máscaras ya combinadas (después de coef @ proto)
#     num_objects = 5
#     mask_h, mask_w = 160, 160  # Resolución típica de máscaras
#     fake_masks = torch.randn(batch_size, num_objects, mask_h, mask_w)
#     fake_features = torch.randn(batch_size, ch[0], 80, 80)
    
#     refined = head.apply_refinement(fake_masks, fake_features)
#     print(f"Máscaras de entrada: {fake_masks.shape}")
#     print(f"Máscaras refinadas: {refined.shape}")
    
#     print("\n" + "=" * 60)
#     print("✅ Todas las pruebas pasaron!")
#     print("=" * 60)

# ==============================================================================
# SegmentEnhanced con UNetProto: Cabeza de Segmentación Mejorada para YOLO
# ==============================================================================
# 
# Mejoras implementadas:
#   A) Más prototipos y coeficientes (nm=64, npr=512 por defecto)
#   B) Generación de prototipos con decodificador UNet (fusión progresiva)
#   C) Módulo de refinamiento no-lineal para pulir las máscaras
#   D) Atención en skip connections para filtrar ruido
#
# Para usar: Registrar en ultralytics/nn/modules/__init__.py y ultralytics/nn/tasks.py
#
# ==============================================================================

# ==============================================================================
# SegmentEnhanced: Cabeza de Segmentación Mejorada para YOLO
# ==============================================================================
#
# Hereda de Detect para compatibilidad total con el loss de YOLO.
#
# Mejoras implementadas:
#   A) Más prototipos (64 vs 32) para mayor capacidad de representación
#   B) Generación de prototipos con decodificador UNet (fusión progresiva)
#   C) Módulo de refinamiento de máscaras (post-procesamiento)
#   D) Atención en skip connections para filtrar ruido
#   E) Atención entre coeficientes y prototipos
#
# ==============================================================================

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv


# ==============================================================================
# MEJORA D: Atención para Skip Connections
# ==============================================================================

class SkipAttention(nn.Module):
    """
    Módulo de atención para skip connections.
    
    Aprende qué partes del skip connection son relevantes dado el contexto
    del feature map que viene de abajo. Útil para filtrar ruido en imágenes médicas.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, skip: torch.Tensor, from_below: torch.Tensor) -> torch.Tensor:
        q = self.query(skip)
        k = self.key(from_below)
        attention = self.gate(q + k)
        return skip * attention


# ==============================================================================
# MEJORA B: Bloque de Decodificación UNet
# ==============================================================================

class UNetDecoderBlock(nn.Module):
    """
    Un bloque del decodificador UNet.
    
    1. Upsample: agranda el feature map que viene de abajo
    2. Concat: lo concatena con el skip connection
    3. Refine: procesa con convoluciones
    """
    
    def __init__(self, c_from_below: int, c_skip: int, c_out: int, use_attention: bool = False):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(c_from_below, c_from_below, kernel_size=2, stride=2, bias=True)
        
        if c_skip != c_from_below:
            self.skip_proj = Conv(c_skip, c_from_below, k=1)
        else:
            self.skip_proj = nn.Identity()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SkipAttention(c_from_below)
        
        self.refine = nn.Sequential(
            Conv(c_from_below * 2, c_out, k=3),
            Conv(c_out, c_out, k=3),
        )
    
    def forward(self, from_below: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.upsample(from_below)
        skip_projected = self.skip_proj(skip)
        
        if upsampled.shape[2:] != skip_projected.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip_projected.shape[2:], mode='bilinear', align_corners=False)
        
        if self.use_attention:
            skip_projected = self.attention(skip_projected, upsampled)
        
        concat = torch.cat([upsampled, skip_projected], dim=1)
        return self.refine(concat)


# ==============================================================================
# MEJORA B: Generador de Prototipos UNet
# ==============================================================================

class UNetProto(nn.Module):
    """
    Genera prototipos usando un decodificador estilo UNet.
    
    Flujo:
        P5 (20×20) → decoder1 (40×40) → decoder2 (80×80) → upsample (160×160) → prototipos
                         ↑                    ↑
                        P4                   P3
    """
    
    def __init__(self, ch: tuple = (256, 512, 1024), c_: int = 256, c2: int = 32, use_attention: bool = True):
        super().__init__()
        
        self.initial_proj = Conv(ch[-1], c_, k=1)
        
        self.decoder1 = UNetDecoderBlock(c_from_below=c_, c_skip=ch[-2], c_out=c_, use_attention=use_attention)
        self.decoder2 = UNetDecoderBlock(c_from_below=c_, c_skip=ch[-3], c_out=c_, use_attention=use_attention)
        
        self.final_upsample = nn.ConvTranspose2d(c_, c_, kernel_size=2, stride=2, bias=True)
        
        self.proto_head = nn.Sequential(
            Conv(c_, c_, k=3),
            Conv(c_, c2, k=1, act=False),
        )
    
    def forward(self, features: list) -> torch.Tensor:
        p3, p4, p5 = features[0], features[1], features[2]
        
        x = self.initial_proj(p5)
        x = self.decoder1(x, p4)
        x = self.decoder2(x, p3)
        x = self.final_upsample(x)
        
        return self.proto_head(x)


# ==============================================================================
# MEJORA B (alternativa): MultiScaleProto
# ==============================================================================

class MultiScaleProto(nn.Module):
    """Genera prototipos concatenando todas las escalas de una vez."""
    
    def __init__(self, ch: tuple, c_: int = 512, c2: int = 64):
        super().__init__()
        
        self.num_scales = len(ch)
        self.scale_projs = nn.ModuleList([Conv(ch[i], c_ // self.num_scales, k=1) for i in range(self.num_scales)])
        
        self.fusion = nn.Sequential(Conv(c_, c_, k=3), Conv(c_, c_, k=3))
        self.upsample = nn.ConvTranspose2d(c_, c_, kernel_size=2, stride=2, bias=True)
        self.refine = nn.Sequential(Conv(c_, c_, k=3), Conv(c_, c2, k=1, act=False))
        
    def forward(self, features: list) -> torch.Tensor:
        target_size = features[0].shape[2:]
        
        aligned_features = []
        for feat, proj in zip(features, self.scale_projs):
            projected = proj(feat)
            if projected.shape[2:] != target_size:
                projected = F.interpolate(projected, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(projected)
        
        fused = torch.cat(aligned_features, dim=1)
        fused = self.fusion(fused)
        fused = self.upsample(fused)
        return self.refine(fused)


# ==============================================================================
# MEJORA C: Refinador de Máscaras
# ==============================================================================

class MaskRefiner(nn.Module):
    """Refina las máscaras después de la combinación lineal de prototipos."""
    
    def __init__(self, nm: int = 64, feat_channels: int = 256, hidden_dim: int = 64):
        super().__init__()
        
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.feat_encoder = Conv(feat_channels, hidden_dim, k=1)
        self.refine_net = nn.Sequential(
            Conv(hidden_dim * 2, hidden_dim, k=3),
            Conv(hidden_dim, hidden_dim, k=3),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        
    def forward(self, coarse_mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        original_shape = coarse_mask.shape
        
        if len(original_shape) == 4 and original_shape[1] > 1:
            B, N, H, W = original_shape
            coarse_mask = coarse_mask.view(B * N, 1, H, W)
            features = features.repeat_interleave(N, dim=0)
        
        if features.shape[2:] != coarse_mask.shape[2:]:
            features = F.interpolate(features, size=coarse_mask.shape[2:], mode='bilinear', align_corners=False)
        
        mask_encoded = self.mask_encoder(coarse_mask)
        feat_encoded = self.feat_encoder(features)
        combined = torch.cat([mask_encoded, feat_encoded], dim=1)
        residual = self.refine_net(combined)
        refined = coarse_mask + residual
        
        if len(original_shape) == 4 and original_shape[1] > 1:
            refined = refined.view(B, N, H, W)
        
        return refined


# ==============================================================================
# MEJORA E: Atención entre Coeficientes y Prototipos (versión simple)
# ==============================================================================

class CoeffProtoAttentionSimple(nn.Module):
    """Versión ligera de atención entre coeficientes y prototipos."""
    
    def __init__(self, nm: int = 64, hidden_dim: int = 64):
        super().__init__()
        
        self.nm = nm
        self.proto_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(start_dim=2),
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
        )
        self.coeff_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, coefficients: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # Coefficients come in as (B, nm, num_anchors) from the caller (already permuted at line 1013)
        B, nm, num_anchors = coefficients.shape

        proto_encoded = self.proto_encoder(prototypes)
        coeff_reshaped = coefficients.permute(0, 2, 1).unsqueeze(-1)
        coeff_encoded = self.coeff_encoder(coeff_reshaped)
        proto_expanded = proto_encoded.unsqueeze(1).expand(-1, num_anchors, -1, -1)

        combined = torch.cat([coeff_encoded, proto_expanded], dim=-1)
        attn_logits = self.attention(combined).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=-1)

        coeff_permuted = coefficients.permute(0, 2, 1)
        refined_permuted = coeff_permuted * (1 + attn_weights * torch.sigmoid(self.gate))

        # Return in (B, nm, num_anchors) format to match input
        return refined_permuted.permute(0, 2, 1)


# ==============================================================================
# MEJORA E: Atención entre Coeficientes y Prototipos (versión completa)
# ==============================================================================

class CoeffProtoAttention(nn.Module):
    """Versión completa con multi-head attention."""
    
    def __init__(self, nm: int = 64, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        self.nm = nm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.coeff_to_query = nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim))
        self.proto_to_key = nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim))
        self.proto_to_value = nn.Sequential(nn.Linear(1, embed_dim), nn.LayerNorm(embed_dim))
        self.proto_pool = nn.AdaptiveAvgPool2d(1)
        self.out_proj = nn.Linear(embed_dim, 1)
        self.scale = self.head_dim ** -0.5
        self.gate = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    
    def forward(self, coefficients: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        B, nm, num_anchors = coefficients.shape
        
        proto_pooled = self.proto_pool(prototypes).view(B, nm)
        keys = self.proto_to_key(proto_pooled.unsqueeze(-1))
        values = self.proto_to_value(proto_pooled.unsqueeze(-1))
        
        coeff_flat = coefficients.permute(0, 2, 1).reshape(B * num_anchors, nm, 1)
        queries = self.coeff_to_query(coeff_flat)
        
        keys = keys.unsqueeze(1).expand(-1, num_anchors, -1, -1).reshape(B * num_anchors, nm, self.embed_dim)
        values = values.unsqueeze(1).expand(-1, num_anchors, -1, -1).reshape(B * num_anchors, nm, self.embed_dim)
        
        queries = queries.view(B * num_anchors, nm, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B * num_anchors, nm, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B * num_anchors, nm, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).reshape(B * num_anchors, nm, self.embed_dim)
        
        refined_flat = self.out_proj(attn_output).squeeze(-1)
        refined = refined_flat.view(B, num_anchors, nm).permute(0, 2, 1)
        
        coeff_orig = coefficients.unsqueeze(-1)
        coeff_ref = refined.unsqueeze(-1)
        gate_input = torch.cat([coeff_orig, coeff_ref], dim=-1)
        gate_weight = self.gate(gate_input).squeeze(-1)
        
        return gate_weight * refined + (1 - gate_weight) * coefficients


# ==============================================================================
# CABEZA PRINCIPAL: SegmentEnhanced (hereda de Detect)
# ==============================================================================

class SegmentEnhanced(Detect):
    """
    Cabeza de segmentación mejorada para YOLO.
    
    HEREDA DE DETECT para compatibilidad total con el loss de YOLO.
    
    Mejoras sobre Segment original:
    - A) Más prototipos (64 vs 32)
    - B) UNetProto: decodificador progresivo estilo UNet
    - C) MaskRefiner: refinamiento de máscaras (post-proceso)
    - D) SkipAttention: atención en skip connections
    - E) CoeffProtoAttention: atención entre coeficientes y prototipos
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 64,
        npr: int = 512,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        use_unet_proto: bool = True,
        use_attention: bool = True,
        use_coeff_attention: bool = True,
        coeff_attention_type: str = "simple",
    ):
        """
        Args:
            nc: Número de clases
            nm: Número de prototipos (64 por defecto, vs 32 original)
            npr: Canales intermedios para prototipos
            reg_max: Bins para DFL
            end2end: Si usar detección end-to-end
            ch: Canales de entrada por escala
            use_unet_proto: Si usar UNetProto (True) o MultiScaleProto (False)
            use_attention: Si usar atención en skip connections de UNetProto
            use_coeff_attention: Si usar atención entre coeficientes y prototipos
            coeff_attention_type: "simple" o "full"
        """
        # Inicializar Detect (la clase padre)
        super().__init__(nc, reg_max, end2end, ch)
        
        self.nm = nm
        self.npr = npr
        
        # =====================================================================
        # MEJORA B: Generador de prototipos (UNet o MultiScale)
        # =====================================================================
        if use_unet_proto:
            self.proto = UNetProto(ch=ch, c_=npr, c2=nm, use_attention=use_attention)
        else:
            self.proto = MultiScaleProto(ch=ch, c_=npr, c2=nm)
        
        # =====================================================================
        # Cabeza de coeficientes de máscara (MEJORA A: más coeficientes)
        # =====================================================================
        c4 = max(ch[0] // 4, nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, nm, 1)) for x in ch
        )
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
        
        # =====================================================================
        # MEJORA E: Atención entre coeficientes y prototipos
        # =====================================================================
        self.use_coeff_attention = use_coeff_attention
        if use_coeff_attention:
            if coeff_attention_type == "simple":
                self.coeff_attention = CoeffProtoAttentionSimple(nm=nm, hidden_dim=64)
            else:
                self.coeff_attention = CoeffProtoAttention(nm=nm, embed_dim=128, num_heads=4)
        else:
            self.coeff_attention = None
        
        # =====================================================================
        # MEJORA C: Refinador de máscaras (para post-procesamiento)
        # =====================================================================
        self.refiner = MaskRefiner(nm=nm, feat_channels=ch[0], hidden_dim=64)

    @property
    def one2many(self):
        """Returns the one-to-many head components."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list) -> tuple | dict:
        """
        Forward pass.
        
        Sigue el mismo formato que Segment original para compatibilidad con el loss.
        """
        # Guardar features originales para el proto
        features_for_proto = [xi.clone() for xi in x]
        
        # Llamar al forward de Detect (genera boxes, scores, etc.)
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        
        # Generar prototipos con nuestro UNetProto mejorado
        proto = self.proto(features_for_proto)
        
        # Agregar proto al diccionario de predicciones
        if isinstance(preds, dict):
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        
        # Aplicar atención de coeficientes si está habilitada
        if self.use_coeff_attention and self.coeff_attention is not None:
            if isinstance(preds, dict):
                if "mask_coefficient" in preds:
                    # preds["mask_coefficient"] tiene shape (B, num_anchors, nm)
                    # coeff_attention espera (B, nm, num_anchors)
                    mc = preds["mask_coefficient"].permute(0, 2, 1)
                    mc_refined = self.coeff_attention(mc, proto)
                    preds["mask_coefficient"] = mc_refined.permute(0, 2, 1)
                elif "one2many" in preds and "mask_coefficient" in preds["one2many"]:
                    mc = preds["one2many"]["mask_coefficient"].permute(0, 2, 1)
                    mc_refined = self.coeff_attention(mc, proto)
                    preds["one2many"]["mask_coefficient"] = mc_refined.permute(0, 2, 1)
        
        if self.training:
            return preds
        
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def forward_head(self, x: list, box_head, cls_head, mask_head) -> dict:
        """
        Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients.
        """
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]
            preds["mask_coefficient"] = torch.cat(
                [mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2
            )
        return preds

    def apply_refinement(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Aplica refinamiento a máscaras ya combinadas (para post-procesamiento).
        
        Args:
            masks: Máscaras combinadas (B, N, H, W)
            features: Features de la imagen para contexto
        
        Returns:
            Máscaras refinadas
        """
        return self.refiner(masks, features)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None
