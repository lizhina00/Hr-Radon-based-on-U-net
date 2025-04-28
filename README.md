# Hr-Radon-based-on-U-net
Deep learning-based radon domain multiple elimination

Code for the problem of Deep learning-based radon domain multiple elimination.

In order to use it:
1. Set the necessary parameters.
2. Run python trainmultiple.py
3. Run python applymultiple.py

Network raining

The sample training dataset has already been prepared at the following file path:
data_simple/train/
data_complex/train/

The validation sets are stored separately at:
data_simple/validation /
data_complex/validation /

The "low" folder under the specified paths contains the input conjugate solution data, while the "high" folder stores the high-resolution Radon-domain data serving as the label data.

The network training is triggered by executing the script trainmultiple.py via command line:
 python trainmultiple.py  

The following section lists some key parameters in trainmultiple.py that require adjustment prior to executing the program:
dim: This denotes the dimensions of the training data. For instance, in the sample dataset, the dimensions for the simplified model are (256, 64), whereas the actual dataset employs (1600, 32). Please note that this parameter must be updated consistently in the configuration file utilsmultiple.py to ensure compatibility.
seismPathT: Path to the training dataset (input files).
faultPathT: Path to the training dataset (labels).
seismPathV: Path to the validation set (input files).
faultPathV: Path to the validation set (labels).
train_ID: Number of input files.
valid_ID: Number of labels.
model: The neural network architecture defined in the imported module unet3multiple.py.

Application of Neural Network Models

Once training finishes, process your data with the trained model by running:
python applymultiple.py

The parameters below need to be adjusted based on your specific task
model=load_model(): Import a previously trained model into your application.
seismPath: Path to the input files.
faultPath: Path to the lables. This dataset is used for comparison with model predictions to evaluate performance. You may remove the associated code if not required for your use case.
filepath3: Path to the output files.
n1, n2: The dimensions of input data.
dk: This parameter specifies the exact filename of the input data.
