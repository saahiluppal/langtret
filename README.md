# Language Translation System
This Repository provides implementation for <strong>Language Translation System</strong> 
in Tensorflow 2 using all best practices
 ### Implementations Included:
 - Transformers
 - Encoder Decoder with Attention
 
 ## Training And Evaluating Transformers
 Dataset is needed in order to train Transformer model. Go grab a dataset of your choice from the pool <a href='http://www.manythings.org/anki/'>here</a>. Extract the dataset and you are ready to train the model.
 #### Training:
 ```bash
 $ python main_transformers.py --path [path to .txt] \
                               --batch [batch size] \
                               --sample [No of lines to train] \
                               --patience [Patience to early stopping] \
                               --epochs [no of epochs]
 ```
 This will split out the model weights and tokenizers for both the languages.
 Model weights can be found in **checkpoints** directory and tokenizers can be found
 named **tok_lang1.subwords** and **tok_lang2.subwords**
#### Inference:
```bash
$ python evaluate_transfomer.py --input_vocab [path to input vocabulary (in this case tok_lang1.subwords)] \
                                --target_vocab [path to target vocabulary] \
                                --checkpoint [path to checkpoint directory (defaults to ./checkpoints/train)]
```
_evaluate_transformer.py_ script is highly customizable, so you can customize it in your own way. Default configuration will ask you for input and will spit out the prediction.

## Training And Evaluating Encoder-Decoder with attention
Similarly, you can grab a dataset <a href='http://www.manythings.org/anki/'>here</a>.
#### Training:
```bash
$ python main_attention.py --path [path to .txt] \
                           --batch [batch size] \
                           --sample [No of lines to train on] \
                           --patience [Patience to early stopping] \
                           --epochs [no of epochs]
```
 This will split out the model weights and tokenizers for both the languages.
 Model weights can be found in **checkpoints** directory and tokenizers can be found
 named **tok_lang1.subwords** and **tok_lang2.subwords**
 #### Inference:
```bash
$ python evaluate_attention.py --input_vocab [path to input vocabulary (in this case tok_lang1.subwords)] \
                                --target_vocab [path to target vocabulary] \
                                --checkpoint [path to checkpoint directory (defaults to ./checkpoints/train)]
```
_evaluate_attention.py_ script is highly customizable, so you can customize it in your own way. Default configuration will ask you for input and will spit out the prediction.
