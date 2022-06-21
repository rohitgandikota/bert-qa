# NLP-based-Question-Answering-using-BERT-model-in-Hugging-Face
This project shows the usage of hugging face framework to answer questions using a deep learning model for NLP called BERT. This work can be adopted and used in many application in NLP like smart assistant or chat-bot or smart information center. </br>

| ==NOTE==: Running the code requires a proper internet connection for downloading the model from huggingface or should manually download all the files into a folder named `bert-large-uncased-whole-word-masking-finetuned-squad` and save the folder in the working directory. Files can be found in the following [**link**](https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main) |

| INSTALLATIONS: Please have the following libraries installed **torch**, **transformers**, **PyPDF2** |

<hr>
To search for an answer to a question from a PDF, use the `searchAnswerPDF.py` code.

To search for an answer to a question from a text, use the `searchAnswerText.py` code. 
<hr>
To use the `searchAnswerPDF.py`, the following parameters have to be tweeked as per your application.
```
# The question that you want to ask
question = 'What is life expectancy of kompsat-3?'
# The full path of the PDF from which you choose to take the context from
pdf_path='D:\\Projects\\bhoonidhi\\kompsat.pdf'
# The parent path of the working directory where the folder containing model files is present
model_path='D:\\Projects\\bhoonidhi\\'
```
<hr>
The code for `searchAnswerText.py` is explained in detail below:

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

Below is an example of the model when asked a question based on the context provided. 
```
Context:
KOMPSAT-3 is a high performance remote sensing satellite, which provides 0.7 m GSD
panchromatic image and 2.8 m GSD multi-spectral image data for various applications.
KOMPSAT-3 was launched into a sun synchronous low Earth orbit on the 18th of May, 2012
and the life time of more than 7 years is expected.

Question:
What is life expectancy of kompsat-3?

Answer:
More than 7 years.
```
