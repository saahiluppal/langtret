# Language Translation System
This Repository provides implementation for <strong>Language Translation System</strong> 
in Tensorflow 2 using all best practices
 ### Implementations Included:
 - Encoder Decoder with Attention
 - Transformers
 
 ## Training And Evaluating Transformers
 Dataset is needed in order to train Transformer model. Go grab a dataset of your choice from the pool <a href='http://www.manythings.org/anki/'>here</a>. Extract the dataset and you are ready to train the model.
 #### Training:
 ```bash
 $ python main_transformers.py --path [path to .txt]
                               --batch [batch size]
                               --sample [No of lines to train]
                               --patience [Patience to early stopping]
                               --epochs [no of epochs]
 ```
 This will split out the model weights and tokenizers for both the languages.
 Model weights can be found in **checkpoints** directory and tokenizers can be found
 named **tok_lang1.subwords** and **tok_lang2.subwords**
