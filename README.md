# NLP-based-Question-Answering-using-BERT-model-in-Hugging-Face
This project shows the usage of hugging face framework to answer questions using a deep learning model for NLP called BERT. This work can be adopted and used in many application in NLP like smart assistant or chat-bot or smart information center. </br>

The code is explained in detail below:

### Importing the huggingface helpers
```
from transformers import BertForQuestionAnswering  
from transformers import BertTokenizer
```
The above lines imports the model itself and the tokenizer algorithm

### Initializing the model and tokenizer
```
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
```
Running the above code requires a proper internet connection for downloading the model from huggingface or should manually download all the files into a folder named `bert-large-uncased-whole-word-masking-finetuned-squad` and save the folder in the working directory. Files can be found in the following [*link*](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main)

### Defining question and context
```
question = 'When was kompsat-3 launched?'
text = '''KOMPSAT-3 is a high performance remote sensing satellite, which provides 0.7 m GSD
panchromatic image and 2.8 m GSD multi-spectral image data for various applications.
KOMPSAT-3 was launched into a sun synchronous low Earth orbit on the 18th of May, 2012
and the life time of more than 7 years is expected.'''

input_ids = tokenizer.encode(question, text)
print("The input has a total of {} tokens.".format(len(input_ids)))
tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):
    print('{:8}{:8,}'.format(token,id))
```
The tokenizer encodes the text into numbers using `encode` function. 

### Running the model
```
output = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))
```
The encoded text is sent to the model as a torch tensor

### Viewing the output
```
answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)
if answer_end >= answer_start:
    answer = " ".join(tokens[answer_start:answer_end+1])
else:
    print("I am unable to find the answer to this question. Can you please ask another question?")
    
print("\nQuestion:\n{}".format(question.capitalize()))
print("\nAnswer:\n{}.".format(answer.capitalize()))
```
