**Secure Authentication Using Facial Recognition Algorithms**

Login app that has an extra security layer using tensorflow transfer learning and flask

* Database used for training = http://www.anefian.com/research/face_reco.htm 


*  For the rest API, I used Flask package from python

* Project was built on M1 mac, so for this mac models do the following steps:
    * Install conda environment
    * Activate conda env. Command: conda activate
    * Install tensorflow-macos from pip. Command: python3 -m pip install tensorflow-macos
    * Install tensorflow-metal for gpu. Command: python3 -m pip install tensorflow-metal
    

* Command to run the app: python3 server.py:
    * To Do: Add shebang line for running with python3 from terminal with command: ./server.py -- might need to change permisions( command for that: chmod +x server.py)

* For transfer learning I used mobilenet v2 model with frozen weights