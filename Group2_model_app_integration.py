from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

import os 

import io
import json
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision import models
from flask import Flask

# init SQLAlchemy so we can use it later in our models
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    dropzone = Dropzone(app)


    app.config['SECRET_KEY'] = 'supersecretkeygoeshere'

    # Dropzone settings
    app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
    app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
    app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
    app.config['DROPZONE_REDIRECT_VIEW'] = 'results'

    # Uploads settingsv
    app.config['UPLOADED_PHOTOS_DEST'] = 'C:/lungapp/abiapp/static/uploads'
    
    photos = UploadSet('photos', IMAGES)
    configure_uploads(app, photos)
    patch_request_class(app)  # set maximum file size, default is 16MB
    app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'

    db.init_app(app)
    
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
          # set session for image results
          #app.config['UPLOADED_PHOTOS_DEST'] = 'C:/Users/Abishai/OneDrive/Desktop/abiapp/static/uploads/test'
          if "file_urls" not in session:
              session['file_urls'] = []
          # list to hold our uploaded image urls
          file_urls = session['file_urls']

          # handle image upload from Dropszone
          if request.method == 'POST':
              file_obj = request.files
              for f in file_obj:
                    file = request.files.get(f)
                    
                    # save the file with to our photos folder
                    filename = photos.save(
                        file,
                        name=file.filename    
                    )

                    # append image urls
                    file_urls.append(photos.url(filename))
                    
              session['file_urls'] = file_urls
              return "uploading..."
          # return dropzone template on GET request
          return render_template('upload.html')

    @app.route('/results')
    def results():
    
          # redirect to home if no images to display
          if "file_urls" not in session or session['file_urls'] == []:
              return redirect(url_for('upload'))
          
          # set the file_urls and remove the session variable
          file_urls = session['file_urls']
          session.pop('file_urls', None)
          
          return render_template('results.html', file_urls=file_urls)
         

    import os 

    import io
    import json
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch import nn
    from torchvision import transforms as T
    from torchvision import models

    @app.route('/detect', methods=['GET', 'POST'])
    def detect():
         files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
         return render_template('detect.html', files_list=files_list)
    
    @app.route('/detectres/<imgg>',methods=['GET','POST'])
    def detectres(imgg):
         idx2label =('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion' , 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Normal Study', 'Pleural_thickening', 'Pneumonia', 'Pneumothorax')
         PATH = 'C:\\lungapp\\lungdiseasemodel.pt'

         num_classes =15
         resnet = models.resnet18(pretrained=False)
         resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
         resnet.fc = nn.Linear(in_features = 512, out_features = num_classes, bias = True)
         resnet.load_state_dict(torch.load(PATH))
         resnet.eval()
         file_url = 'C:\\lungapp\\abiapp\\static\\uploads\\'+imgg
         image=open(file_url, 'rb').read()
         target_size=(224, 224)
         image = Image.open(io.BytesIO(image))
         image = T.Resize(target_size)(image)
         image = T.ToTensor()(image)

         # Convert to Torch.Tensor and normalize.
         image = T.Normalize([0.449], [0.226])(image)

         # Add batch_size axis.
         image = image[None]
         ####if use_gpu:
         ####image = image.cuda()
         image=torch.autograd.Variable(image, volatile=True)
         data = {"success": False}
         outputs = resnet(image)
         _, pred = torch.max(outputs.data, 1)

         data['predictions'] = list()
         r = ''.join('%5s' % idx2label[pred])
         data=r

         ress=data
         ress = ''.join(str(e) for e in ress)
         print(ress)
         return render_template("detectres.html",ress = ress)
    
    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
          # since the user_id is just the primary key of our user table, use it in the query for the user
          return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app