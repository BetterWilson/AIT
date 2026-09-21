from functools import partial  # 偏函数工具，用于把 tokenizer 绑定进 collate_fn
from torch.utils.data import DataLoader  # 批量加载器


def collate_fct(batch, tokenizer):
    """将一批 BPE 句对整理为 encoder/decoder 输入与标签。"""
    src_words = [pair[0].split() for pair in batch]  # 源语切成 token 列表
    trg_words = [pair[1].split() for pair in batch]  # 目标语切成 token 列表

    encoder_inputs, encoder_inputs_mask = tokenizer.encode(  # 源语编码(含 BOS/EOS)
        src_words, padding_first=False, add_bos=True, add_eos=True, return_mask=True
    )
    decoder_inputs = tokenizer.encode(  # decoder 输入: 含 BOS、不含 EOS(相当于右移一位)
        trg_words, padding_first=False, add_bos=True, add_eos=False, return_mask=False
    )
    decoder_labels, decoder_labels_mask = tokenizer.encode(  # decoder 标签: 不含 BOS、含 EOS
        trg_words, padding_first=False, add_bos=False, add_eos=True, return_mask=True
    )

    return {  # 返回一个 batch 的所有张量
        "encoder_inputs": encoder_inputs,  # 编码器输入
        "encoder_inputs_mask": encoder_inputs_mask,  # 编码器掩码(1=PAD)
        "decoder_inputs": decoder_inputs,  # 解码器输入
        "decoder_labels": decoder_labels,  # 解码器标签
        "decoder_labels_mask": decoder_labels_mask,  # 解码器标签掩码(1=PAD)
    }


# 把 tokenizer 绑定进 collate_fn，供 DataLoader 组批时调用
collate_fn_with_tokenizer = partial(collate_fct, tokenizer=tokenizer)  # 偏函数
train_dl = batch_sampler.build_dataloader(collate_fn_with_tokenizer)  # 构建训练集 DataLoader
for i, batch in enumerate(train_dl):  # 取前两个 batch 检查各张量形状
    print(f"encoder_inputs: {batch['encoder_inputs'].shape}")  # 编码器输入形状
    print(f"encoder_inputs_mask: {batch['encoder_inputs_mask'].shape}")  # 编码器掩码形状
    print(f"decoder_inputs: {batch['decoder_inputs'].shape}")  # 解码器输入形状
    print(f"decoder_labels: {batch['decoder_labels'].shape}")  # 解码器标签形状
    print(f"decoder_labels_mask: {batch['decoder_labels_mask'].shape}")  # 标签掩码形状
    if i == 1:  # 看完两个 batch 即停止
        break

# 取出一个 batch 备用，供后续模型搭建章节做形状测试
encoder_inputs = batch["encoder_inputs"]  # 编码器输入
encoder_inputs_mask = batch["encoder_inputs_mask"]  # 编码器输入掩码(1=PAD)
decoder_inputs = batch["decoder_inputs"]  # 解码器输入
decoder_labels = batch["decoder_labels"]  # 解码器标签
decoder_labels_mask = batch["decoder_labels_mask"]  # 解码器标签掩码(1=PAD)
