from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
token_list = tokenizer.encode("How old are you, The Eiffel Tower is in Paris", add_special_tokens=True)
input_ids = torch.tensor(token_list).unsqueeze(0)
segments_ids = [0] * (token_list.index(1010) + 1) + [1] * (len(token_list) - token_list.index(1010) - 1)
segments_tensors = torch.tensor([segments_ids])
outputs = model(input_ids, token_type_ids=segments_tensors)
seq_relationship_scores = outputs[0]
print (seq_relationship_scores)
softmax = torch.nn.Softmax(dim=1)
prediction = softmax(seq_relationship_scores)
print (prediction)
