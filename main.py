import os
import keras
import cv2
import numpy as np
from PIL import Image
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from keras.datasets import cifar10
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


##class name dictionary
dic = {1:'airplane',2:'automobile',3:'bird',4:'cat',5:'deer',6:'dog',7:'frog',8:'horse',9:'ship',10:'truck'}
cat = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
UPLOAD_FOLDER = '/home/manan/Documents/sem7/dl/flask/static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file():
    print("al least here")
    if request.method == 'POST':
        # check if the post request has the file pa
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return  filename

def image_resize(image):
    
    resized = cv2.resize(image, (32,32))
    
    return resized         

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='Image Classification')


@app.route('/clas.html', methods=['GET', 'POST'])
def clas():
    fpath = upload_file() 
    path =os.path.join('static/images/',fpath) 
    img = Image.open(path)
    img = img.resize((32,32))
    image_1 = np.array(img)
    print(path)
    print(img)
    # image_1 = image_resize(img)
    
    data = image_1.reshape(1,32,32,3)
    datan = data/255.0
    model = load_model('models/c10_cnn_man.h5')
    pred_c = model.predict_classes(datan)
    print(pred_c[0]+1)
    pred = dic[pred_c[0]+1]
    print(pred)
    K.clear_session()
    return render_template('clas.html', the_title='Cifar-10 Classification',image=fpath,predict =pred )


@app.route('/tran.html', methods=['GET', 'POST'])
def tran():
    fpath = request.form.get('fname')
    path =os.path.join('static/images/',fpath) 
    img = Image.open(path)
    img = img.resize((64,64))
    image_1 = np.array(img)
    print(path)
    print(img)
    # image_1 = image_resize(img)
    image = (image_1 - 122.15242115234375)/ (1e-7+ 66.52826186369329)
    data = image.reshape(1,64,64,3)

    model1 = load_model('models/cifar10_vgg16.h5')
    pred = model1.predict(data)
    pred_name = cat[np.argmax(pred)]
    print(pred_name)
    K.clear_session()
    return render_template('tran.html', the_title='Cifar-10 Classification(transfer learning vgg16)',image=fpath,predict =pred_name )

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()





