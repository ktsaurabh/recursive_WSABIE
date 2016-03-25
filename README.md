# Recursive WSABIE (to be updated)

This program depends upon keras and theano. So make sure to install these with appropriate path provided.  

./data folder contains a sample file to run the program. These are 1000 abstracts obtained from CiteSeer.
./models folder will hold the model learned.

GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python tag_embeddings_test_rsdae.py
        
CPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=cpu python tag_embeddings_test_rsdae.py