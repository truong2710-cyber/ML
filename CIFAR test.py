import numpy as np
import matplotlib.pyplot as plt
import pickle

cifar10_dir = 'D:/Python/ML/cifar-10-batches-py'
# load flattened image vectors for training and testing
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
        
    features = batch['data'].reshape((len(batch['data']), -1))
    labels = batch['labels']
        
    return features, np.array(labels)

X_train=np.concatenate((load_cfar10_batch(cifar10_dir, 1)[0],load_cfar10_batch(cifar10_dir, 2)[0],load_cfar10_batch(cifar10_dir, 3)[0],load_cfar10_batch(cifar10_dir, 4)[0],load_cfar10_batch(cifar10_dir, 5)[0]),axis=0)
y_train=np.concatenate((load_cfar10_batch(cifar10_dir, 1)[1],load_cfar10_batch(cifar10_dir, 2)[1],load_cfar10_batch(cifar10_dir, 3)[1],load_cfar10_batch(cifar10_dir, 4)[1],load_cfar10_batch(cifar10_dir, 5)[1]),axis=0)
X_test,y_test=load_cfar10_batch(cifar10_dir, 6)

num_training=49000
num_validation=1000
num_test=1000
num_dev=500

mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask].T

X_train = X_train[0:num_training]
y_train = y_train[0:num_training].T

mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask].T


X_test = X_test[0:num_test]
y_test = y_test[0:num_test].T

# Preprocessing: subtract the mean image
mean_image=np.mean(X_train, axis=0).astype(int)
X_train = X_train- mean_image
X_val = X_val- mean_image
X_test = X_test - mean_image
X_dev = X_dev - mean_image
# using bias trick
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))]).T

def svm_loss_vectorized(W,X,y,reg):
    d,C=W.shape
    d,N=X.shape
    loss=0
    dW=np.zeros(W.shape)
    Z=W.T.dot(X)
    correct_class_score=np.choose(y,Z).reshape(1,N) # value of scores of the correct classes
    margins = np.maximum(0, Z - correct_class_score + 1) # loss matrix
    margins[y,np.arange(N)]=0 # set loss at correct classes = 0 
    loss=np.sum(margins)
    loss=loss/N+reg/2*np.linalg.norm(W)
    F = (margins > 0).astype(int) 
    F[y, np.arange(F.shape[1])] = np.sum(-F, axis = 0)
    dW = X.dot(F.T)/N + reg*W
    return loss, dW

def multiclass_svm_minibatchGD(X,y,Winit,reg,eta,batch_size=100,num_iter=5000,print_after=100):
    W=Winit
    for i in range(num_iter):
        idx = np.random.choice(X.shape[1], batch_size)
        X_batch = X[:, idx]
        y_batch = y[idx]
        W-=eta*svm_loss_vectorized(W,X_batch,y_batch,reg)[1]
#        if i%print_after==0:
#            print("Loss at {}/{} iter: {}".format(i,num_iter,svm_loss_vectorized(W,X_batch,y_batch,reg)[0]))
    
    return W

C=10
d=X_train.shape[0]
Winit = np.random.randn(d, C)
#W=multiclass_svm_minibatchGD(X_train,y_train,Winit,reg=1e+4,eta=3e-7)
#y_train_pred=np.argmax(W.T.dot(X_train),axis=0)
#print('Training accuracy: %f' % (np.mean(y_train == y_train_pred)))
#y_val_pred=np.argmax(W.T.dot(X_val),axis=0)
#print('Validation accuracy: %f' % (np.mean(y_val == y_val_pred)))
etas = [1e-7, 2e-7, 3e-7, 8e-7]                      # learning rates
regs = [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 1e5] # regularization_strengths
results = {}
best_val = -1 
max_eta_reg=(0,0)
for eta in etas:
    for reg in regs:
        W=multiclass_svm_minibatchGD(X_train,y_train,Winit,reg,eta)
        y_train_pred=np.argmax(W.T.dot(X_train),axis=0)
        y_val_pred=np.argmax(W.T.dot(X_val),axis=0)
        acc_train=np.mean(y_train == y_train_pred)
        acc_val=np.mean(y_val == y_val_pred)
        results[(eta,reg)]=(acc_train,acc_val)
        if acc_val>best_val:
            best_val=acc_val
            max_eta_reg=(eta,reg)
print("Best validation accuracy: {} with eta = {} and reg = {}".format(best_val,max_eta_reg[0],max_eta_reg[1]))
# we find that eta=1e-7 and reg=2e4 is the best choice
# then calculate test result and test accuracy
W=multiclass_svm_minibatchGD(X_train,y_train,Winit,reg=2e4,eta=1e-7)
y_test_pred = np.argmax(W.T.dot(X_test),axis=0)
acc_test=np.mean(y_test == y_test_pred)
print("Test accuracy: {}".format(acc_test))