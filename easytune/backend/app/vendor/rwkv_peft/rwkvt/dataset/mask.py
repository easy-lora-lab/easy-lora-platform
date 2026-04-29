import torch
import numpy as np

IGNORE_INDEX = -100

# def create_mask(seq, token1, token2, min_len):
#     # 找到所有特殊标记的索引
#     indices1 = []
#     for i in range(min_len - len(token1) + 1):
#         if np.array_equal(seq[i:i + len(token1)], token1):
#             indices1.append(i)
#     indices2 = []

#     for i in range(min_len - len(token2) + 1):
#         if np.array_equal(seq[i:i + len(token2)], token2):
#             indices2.append(i)
#     mask = torch.zeros(seq.shape)
#     # assert len(indices2)!=0 and len(indices1)!=0
#     select = 0
#     for i in range(min_len):
#         if i in indices1:
#             select = 0
#         elif i in indices2:
#             select = 1
#         mask[i] = select
#     if torch.sum(mask) == 0:
#         mask[:min_len - 1] = 1
#     return mask[1:]

# def create_mask(seq, start, end, min_len):
#     # 找到所有特殊标记的索引
#     for token in seq:
        
#     indices1 = []
#     for i in range(min_len - len(token1) + 1):
#         if np.array_equal(seq[i:i + len(token1)], token1):
#             indices1.append(i)
#     indices2 = []

#     for i in range(min_len - len(token2) + 1):
#         if np.array_equal(seq[i:i + len(token2)], token2):
#             indices2.append(i)
#     mask = torch.zeros(seq.shape)
#     # assert len(indices2)!=0 and len(indices1)!=0
#     select = 0
#     for i in range(min_len):
#         if i in indices1:
#             select = 0
#         elif i in indices2:
#             select = 1
#         mask[i] = select
#     if torch.sum(mask) == 0:
#         mask[:min_len - 1] = 1
#     return mask[1:]

# def generate_mask(seq, token1, token2, min_len):
#     mask = torch.zeros(seq.shape)  # 初始化mask列表，默认全为0
#     current_mask_value = 0  # 初始状态下，所有位置的mask值为0

#     i = 0
#     while i < min_len:
#         if seq[i:i + len(token1)] == token1:
#             current_mask_value = 0
#             for j in range(len(token1)):
#                 mask[i + j] = current_mask_value
#             i += len(token1)
#         elif seq[i:i + len(token2)] == token2:
#             current_mask_value = 1
#             for j in range(len(token2)):
#                 mask[i + j] = current_mask_value
#             i += len(token2)
#         else:
#             mask[i] = current_mask_value
#             i += 1

#     if torch.sum(mask) == 0:
#         mask[:min_len - 1] = 1
#     return mask[1:]
def _to_list(seq):
    if isinstance(seq, torch.Tensor):
        return seq.tolist()
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return list(seq)


def _find_subsequence_positions(seq, pattern, limit):
    if not pattern or limit < len(pattern):
        return []
    positions = []
    for i in range(limit - len(pattern) + 1):
        if seq[i:i + len(pattern)] == pattern:
            positions.append(i)
    return positions


def create_mask(seq, token1, token2, min_len):
    """Keep next-token labels only for assistant spans and mask everything else."""
    seq_list = _to_list(seq)
    upper = min(min_len, len(seq_list))
    labels = torch.full((max(len(seq_list) - 1, 0),), IGNORE_INDEX, dtype=torch.long)
    if upper <= 1:
        return labels

    user_pattern = _to_list(token1)
    assistant_pattern = _to_list(token2)
    markers = []
    for pos in _find_subsequence_positions(seq_list, user_pattern, upper):
        markers.append((pos, "user", len(user_pattern)))
    for pos in _find_subsequence_positions(seq_list, assistant_pattern, upper):
        markers.append((pos, "assistant", len(assistant_pattern)))
    markers.sort(key=lambda item: item[0])

    found_assistant = False
    for idx, (start, role, marker_len) in enumerate(markers):
        if role != "assistant":
            continue
        found_assistant = True
        content_start = max(start + marker_len, 1)
        content_end = markers[idx + 1][0] if idx + 1 < len(markers) else upper
        content_end = min(content_end, upper)
        if content_start < content_end:
            labels[content_start - 1:content_end - 1] = torch.tensor(
                seq_list[content_start:content_end], dtype=torch.long
            )

    if not found_assistant:
        labels[:upper - 1] = torch.tensor(seq_list[1:upper], dtype=torch.long)

    return labels


def generate_mask(seq, token1, token2, min_len):
    return create_mask(seq, token1, token2, min_len)
mask_fn_dict = {
    "qa": create_mask,
    "se": generate_mask
}
