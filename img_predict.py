import numpy as np
from keras.models import load_model
from keras.preprocessing import image

mymodel=load_model('mymodel6.h5')

# img without mask
test_image = image.load_img('train/with_mask/2-with-mask.jpg', target_size=(150,150,3))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
pred = mymodel.predict(test_image)[0][0]
if(pred == 1):
    print('no mask\n')
else:
    print('mask\n')

# img with mask
test_image2=image.load_img('train/without_mask/3.jpg', target_size=(150,150,3))
test_image2=image.img_to_array(test_image2)
test_image2=np.expand_dims(test_image2,axis=0)
pred = mymodel.predict(test_image2)[0][0]
if(pred == 1):
    print('no mask\n')
else:
    print('mask\n')