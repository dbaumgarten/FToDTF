[![Build Status](https://travis-ci.org/dbaumgarten/FToDTF.svg?branch=master)](https://travis-ci.org/dbaumgarten/FToDTF)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3872f2d4f965425ea150abd921027f4c)](https://www.codacy.com/app/incognym/FToDTF?utm_source=github.com&utm_medium=referral&utm_content=dbaumgarten/FToDTF&utm_campaign=Badge_Coverage)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3872f2d4f965425ea150abd921027f4c)](https://www.codacy.com/app/incognym/FToDTF?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dbaumgarten/FToDTF&amp;utm_campaign=Badge_Grade)
# FToDTF

Todo: Project description

## Installation - Developers
- clone the repository
- go to the cloned repository
- run ```sudo pip3 install -e .```  
The programm is now installed system-wide. You can now import the package ftodtf in python3 and run the cli-command ```fasttext <optional args>```
- Because you specified ```-e``` when running ```pip3 install``` you can modify the project files and your installation will still always be up-to-date (symlink-magic!).

## Running
After installing just run  
```
fasttext preprocess --corpus_path <your-training-data>
fasttext train
```  
in your console

## Docker
This application is also available as pre-built docker-image (https://hub.docker.com/r/dbaumgarten/ftodtf/)
```
sudo docker run --rm -it -v `pwd`:/data dbaumgarten/ftodtf train
```

## Distributed Setup
There is docker-compose file demonstrating the distributed setup op this programm. To run a cluster on your local machine 
- go to the directory of the docker-compose file
- preprocess your data using `fasttext preprocess --corpus_path <your-training-data>`
- run:
```
sudo docker-compose up
```
This will start a cluster consisting of two workers and two parameter servers on your machine.  
Each time you restart the cluster it will continue to work from the last checkpoint. If you want to start from zero delete the contents of ./log/distributed on the server of worker0
Please note that running a cluster on a single machine is slower then running a single instance directly on this machine. To see some speedup you will need to use multiple independent machines.

## Known Bugs and Limitations
- When supplying input-text that does not contain sentences (but instead just a bunch of words without punctuation) ```fasttext preprocess``` will hang indefinetly.

## Documentation
You can find the auto-genrated documentation for the code here: https://dbaumgarten.github.io/FToDTF/  
The architecture documentation (german only) can be found here: https://github.com/dbaumgarten/FToDTF/blob/master/docs/architecture/architecture.md