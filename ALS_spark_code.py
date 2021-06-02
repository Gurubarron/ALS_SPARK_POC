Get the data here http://grouplens.org/datasets/movielens/
movielens = sc.textFile("../in/ml-100k/u.data")

movielens.first() #u'196\t242\t3\t881250949'
movielens.count() #100000

#Clean up the data by splitting it
#Movielens readme says the data is split by tabs and
#is user product rating timestamp
clean_data = movielens.map(lambda x:x.split('\t'))

#As an example, extract just the ratings to its own RDD
#rate.first() is 3
rate = clean_data.map(lambda y: int(y[2]))
rate.mean() #Avg rating is 3.52986

#Extract just the users
users = clean_data.map(lambda y: int(y[0]))
users.distinct().count() #943 users

#You don't have to extract data to its own RDD
#This command counts the distinct movies
#There are 1,682 movies
clean_data.map(lambda y: int(y[1])).distinct().count() 

#Need to import three functions / objects from the MLlib
from pyspark.mllib.recommendation\
    import ALS,MatrixFactorizationModel, Rating

#We'll need to map the movielens data to a Ratings object 
#A Ratings object is made up of (user, item, rating)
mls = movielens.map(lambda l: l.split('\t'))
ratings = mls.map(lambda x: Rating(int(x[0]),\
    int(x[1]), float(x[2])))

#Need a training and test set
train, test = ratings.randomSplit([0.7,0.3],7856)

train.count() #70,005
test.count() #29,995

#Need to cache the data to speed up training
train.cache()
test.cache()

#Setting up the parameters for ALS
rank = 5 # Latent Factors to be made
numIterations = 10 # Times to repeat process
#Create the model on the training data
model = ALS.train(train, rank, numIterations)

#Examine the latent features for one product
model.productFeatures().first()
#(12, array('d', [-0.29417645931243896, 1.8341970443725586, 
    #-0.4908868968486786, 0.807500958442688, -0.8945541977882385]))

#Examine the latent features for one user
model.userFeatures().first()
#(12, array('d', [1.1348751783370972, 2.397622585296631,
    #-0.9957215785980225, 1.062819480895996, 0.4373367130756378]))

# For Product X, Find N Users to Sell To
model.recommendUsers(242,100)

# For User Y Find N Products to Promote
model.recommendProducts(196,10)

#Predict Single Product for Single User
model.predict(196, 242)

# Predict Multi Users and Multi Products
# Pre-Processing
pred_input = train.map(lambda x:(x[0],x[1]))   

# Lots of Predictions
#Returns Ratings(user, item, prediction)
pred = model.predictAll(pred_input) 

#Get Performance Estimate
#Organize the data to make (user, product) the key)
true_reorg = train.map(lambda x:((x[0],x[1]), x[2]))
pred_reorg = pred.map(lambda x:((x[0],x[1]), x[2]))

#Do the actual join
true_pred = true_reorg.join(pred_reorg)

#Need to be able to square root the Mean-Squared Error
from math import sqrt

MSE = true_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
RMSE = sqrt(MSE)#Results in 0.7629908117414474

#Test Set Evaluation
#More dense, but nothing we haven't done before
test_input = test.map(lambda x:(x[0],x[1])) 
pred_test = model.predictAll(test_input)
test_reorg = test.map(lambda x:((x[0],x[1]), x[2]))
pred_reorg = pred_test.map(lambda x:((x[0],x[1]), x[2]))
test_pred = test_reorg.join(pred_reorg)
test_MSE = test_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
test_RMSE = sqrt(test_MSE)#1.0145549956596238

#If you're happy, save your model!
model.save(sc,"../out/ml-model")