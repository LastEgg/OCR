import torch
from models import Huggingface_LAST_ONNX, TrOCRConfig_ONNX
import onnx
import onnxruntime as ort

def huggingface_last_load_kvcache(model_path):
    model = Huggingface_LAST_ONNX(
            config=TrOCRConfig_ONNX(vocab_size=415, max_length=100, d_model=256,
            decoder_layers=3,
            decoder_attention_heads=8,
            decoder_ffn_dim=1024, 
            max_position_embeddings=100,
            dropout=0.3,
            # use_cache=True,
            ),
            nline=16,
            num_layers=16,
            growth_rate=24,
            )
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    update_dict = {}
    for k, v in state_dict.items():
        update_dict[k[6:]] = v
    msg = model.load_state_dict(update_dict)
    print(f"模型加载：{msg}")
    model.eval()
    return model

def export_encoder(model):
    # 修改 forward 方法以提供默认参数
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, image, image_mask):
            return self.encoder(image, image_mask)

    wrapped_encoder = EncoderWrapper(model.encoder)

    # 准备输入
    dummy_image = torch.rand(1, 3, 256, 256)  # 假设图像输入形状
    dummy_mask  = torch.zeros((256, 256), dtype=torch.bool).unsqueeze(0) # 假设图像输入形状
    onnx_model_path = "./checkpoints/onnx/encoder.onnx"

    # 导出模型
    torch.onnx.export(
        wrapped_encoder,
        (dummy_image,dummy_mask),
        onnx_model_path,
        input_names=["image", "image_mask"],
        output_names=["output", "mask"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "image_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
            "mask": {0: "batch_size"}
        },
        opset_version=12
    )

    print(f"Model exported to {onnx_model_path}")

def export_decoder(model):
    # 修改 forward 方法以提供默认参数
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, encoder_input, task_seq_input,
                    task_seq_pos, task_seq_li
                    ): 
            output = self.decoder(
                encoder_input, 
                task_seq_input,
                task_seq_pos, 
                task_seq_li, 
                past_key_values=None,
            )
            return output.logits, output.past_key_values

    wrapped_decoder = DecoderWrapper(model.decoder)

    # 准备输入
    image_width = 256
    image_height = 256
    max_tokens = 256
    decoder_layers = model.config.decoder_layers
    decoder_attention_heads = model.config.decoder_attention_heads
    d_model = model.config.d_model
    head_dim = d_model // decoder_attention_heads


    dummy_encoder_hidden_states = torch.rand(1, max_tokens, d_model)  #编码器输出形状
    dummy_decoder_input_ids = torch.zeros((1, max_tokens), dtype=torch.long)  #解码器输入ID
    dummy_decoder_input_pose = torch.rand(1, max_tokens, d_model)
    dummy_decoder_input_li = torch.zeros((1, max_tokens), dtype=torch.long)  

    onnx_model_path = "./checkpoints/onnx/decoder.onnx"

    # 导出模型
    torch.onnx.export(
        wrapped_decoder,
        (
            dummy_encoder_hidden_states, 
            dummy_decoder_input_ids,
            dummy_decoder_input_pose,
            dummy_decoder_input_li, 
        ),
        onnx_model_path,
        input_names=["encoder_input", "task_seq_input","task_seq_pos", "task_seq_li"],
        output_names=["output", "kvcache"],
        dynamic_axes={
            "encoder_input": {0: "batch_size", 1: "encoder_sequence_length"},
            "task_seq_input": {0: "batch_size", 1: "sequence_length"},
            "task_seq_pos": {0: "batch_size", 1: "sequence_length"},
            "task_seq_li": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size", 1: "sequence_length"}

        },
        opset_version=12
    )

    print(f"Model saved to: {onnx_model_path}")

    model = onnx.load(onnx_model_path)
    for input in model.graph.input:
        print("@@@", input.name)

def export_decoder_with_kvcache(model):
    # 修改 forward 方法以提供默认参数
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, encoder_input, task_seq_input,
                    task_seq_pos, task_seq_li,
                    past_key_values
                    ): 
            output = self.decoder(
                encoder_input, 
                task_seq_input,
                task_seq_pos, 
                task_seq_li, 
                past_key_values,
            )
            return output.logits, output.past_key_values

    wrapped_decoder = DecoderWrapper(model.decoder)

    # 准备输入
    image_width = 256
    image_height = 256
    max_tokens = 256
    decoder_layers = model.config.decoder_layers
    decoder_attention_heads = model.config.decoder_attention_heads
    d_model = model.config.d_model
    head_dim = d_model // decoder_attention_heads


    dummy_encoder_hidden_states = torch.rand(1, max_tokens, d_model)  #编码器输出形状
    dummy_decoder_input_ids = torch.zeros((1, max_tokens), dtype=torch.long)  #解码器输入ID
    dummy_decoder_input_pose = torch.rand(1, max_tokens, d_model)
    dummy_decoder_input_li = torch.zeros((1, max_tokens), dtype=torch.long)  

    past_key_values = tuple(
        (
            torch.zeros(1, decoder_attention_heads, 1, head_dim),
            torch.zeros(1, decoder_attention_heads, 1, head_dim),
            torch.zeros(1, decoder_attention_heads, max_tokens, head_dim),
            torch.zeros(1, decoder_attention_heads, max_tokens, head_dim)
        )
        for _ in range(decoder_layers * 4)
    )

    onnx_model_path = "./checkpoints/onnx/decoder_kvcache.onnx"

    # 导出模型
    torch.onnx.export(
        wrapped_decoder,
        (
            dummy_encoder_hidden_states, 
            dummy_decoder_input_ids,
            dummy_decoder_input_pose,
            dummy_decoder_input_li, 
            past_key_values
        ),
        onnx_model_path,
        input_names=["encoder_input", "task_seq_input","task_seq_pos", "task_seq_li"] + ['kvcache_' + str(i) for i in range(decoder_layers * 4)],
        output_names=["output", "kvcache"],
        dynamic_axes={
            "encoder_input": {0: "batch_size", 1: "encoder_sequence_length"},
            "task_seq_input": {0: "batch_size", 1: "sequence_length"},
            "task_seq_pos": {0: "batch_size", 1: "sequence_length"},
            "task_seq_li": {0: "batch_size", 1: "sequence_length"},
            **{f'kvcache_{i}': {0: 'batch_size', 2: 'past_sequence_length'} for i in range(decoder_layers * 4)},
            "output": {0: "batch_size", 1: "sequence_length"}

        },
        opset_version=12
    )

    print(f"Model saved to: {onnx_model_path}")

    model = onnx.load(onnx_model_path)
    for input in model.graph.input:
        print("@@@", input.name)

if __name__ == '__main__':
    model = huggingface_last_load_kvcache("/datassd/hz/gdx_ocr/checkpoints/baseline_LAST_V1.1/epoch_36-loss_0.367.ckpt")
    export_encoder(model)
    export_decoder_with_kvcache(model)
    export_decoder(model)

