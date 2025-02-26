# 原版的LAST源码
from .LAST.last import LAST
# 用于kv cache版本的训练
from .HG_LAST.huggingface_last import Huggingface_LAST 
from .HG_LAST.transformers import TrOCRConfig
# 用于kv cache版本的推理
from .KV_LAST.huggingface_last import Huggingface_LAST as Huggingface_LAST_infer
from .KV_LAST.huggingface_last import TrOCRConfig as TrOCRConfig_infer
# 用于ONNX的导出
from .KV_LAST_ONNX.huggingface_last import Huggingface_LAST as Huggingface_LAST_ONNX
from .KV_LAST_ONNX.huggingface_last import TrOCRConfig as TrOCRConfig_ONNX