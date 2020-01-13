import numpy as np

class SVDPP:

    def __init__(self,R,hidden_dim = 100,weight_scale = 1e-3,dtype=np.float32):
        
        self.num_users, self.num_items = R.shape
        self.params = {}
        self.params['bu'] = np.zeros([self.num_users])
        self.params['bi'] = np.zeros([self.num_items])
        self.params['Wu'] = np.random.normal(0,weight_scale,[self.num_users,hidden_dim])
        self.params['Wi'] = np.random.normal(0,weight_scale,[hidden_dim,self.num_items])
        self.params['y'] = np.random.normal(0,weight_scale,[self.num_items, hidden_dim])
        self.impli = (R>0).astype(np.float32)
        for i,u in enumerate(self.impli):
            self.impli[i] = u/np.sqrt(u.sum())
        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def forward(self,R,mask,reg):
        # R is the normalized rating matrix whose shape is (num_users, num_items)
        # return loss, cache
        loss = 0
        error = (self.params['Wu']+self.impli.dot(self.params['y'])).dot(self.params['Wi'])+self.params['bi']-R  # num_items, num_users
        error = (error.T+self.params['bu']).T
        error *= mask
        num_train = float(mask.sum())
        loss = 0.5*(error*error).sum()/num_train
        loss += 0.5*reg*( np.sum(self.params['Wu']**2)+np.sum(self.params['Wi']**2)
                            +np.sum(self.params['bu']**2)+np.sum(self.params['bi']**2) 
                           +np.sum(self.params['y']**2)   )
        cache = ( error,reg,num_train )
        return loss, cache
            
    def backward(self,cache):
        error,reg,num_train = cache
        dbu = error.sum(axis=1)/num_train + reg*self.params['bu']
        dbi = error.sum(axis=0)/num_train + reg*self.params['bi']
        tmp = error.dot(self.params['Wi'].T)/num_train
        dWu = tmp + reg*self.params['Wu']
        dWi = self.params['Wu'].T.dot(error)/num_train + reg*self.params['Wi']
        dy = self.impli.T.dot(tmp) + reg*self.params['y']
        
        return dWu,dWi,dbu,dbi,dy
        
    def train(self,R,mask,epochs,lr=1e-3,reg=1e-3):
        loss_history = []
        
        for i in range(epochs):
            cur_loss,cache = self.forward(R,mask,reg)
            loss_history.append(cur_loss)
            grads = {}
            grads['Wu'],grads['Wi'],grads['bu'],grads['bi'],grads['y'] = self.backward(cache)
            for k,v in self.params.items():
                self.params[k] -= lr*grads[k]
        return loss_history
    
    def test(self,Rtest):  
        # test set shape :   test record   rating   return a matrix of shape (num_users, 20)
        # user_id  movie_id  rating  timestamp
        predict = self.params['Wu'].dot(self.params['Wi'])+self.params['bi']
        predict = (predict.T+self.params['bu']).T
        
        rec_lists = [None]*(self.num_users+1)
        see_users =[0]*(self.num_users+1)
        for record in Rtest:
            uid, mid, _, _ = record
            if see_users[uid]==0:
                see_users[uid]=1
                rec_lists[uid]=[]
            
            tup = ( mid, predict[uid-1][mid-1] )
            rec_lists[uid].append(tup)
            
        for reclist in rec_lists:
            if reclist is not None:
                reclist.sort(key=lambda x:x[1],reverse=True)
                
        return rec_lists
    
    def test_bi(self):
        predict = (self.params['Wu']+self.impli.dot(self.params['y']) ).dot(self.params['Wi'])+self.params['bi']
        predict = (predict.T+self.params['bu']).T
        rec_lists = [None]*(self.num_users+1)
        for i in range(self.num_users):
            mydict = {k+1:v for k,v in enumerate(predict[i])}
            rec_lists[i+1] = sorted(mydict.items(), key=lambda x: x[1], reverse=True)
            
        return rec_lists

