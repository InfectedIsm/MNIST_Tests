import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #mettre a '0' pour activer les warnings

import tensorflow as tf
import numpy as np
#import matplotlib as mp

#importation des données d'apprentissage et test depuis un ftp
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True )	
print("\nextraction completed")
print("Note that you can modify some parameters in the code")
print("number_of_training, training_set_size, hidden_layer_width, alpha")

print("\n... Now please wait, dependings on parameters values, it can take a while ...")

number_of_training = 200
training_set_size = 10

#L'entrée est un placeholder car on la feed une seule fois au début
#None veut dire que l'on ne fixe pas de nombre de lignes  
##j'ai ajouté une hidden layer, de taille réglable
hidden_layer_width = 100

tf.reset_default_graph() 
with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32, [None, 784],name='data')
	y_ = tf.placeholder(tf.float32, [None, 10], name='labels')
		
with tf.name_scope('1st_layer'):
	with tf.name_scope('weights'):
		W1 = tf.Variable(tf.truncated_normal([784,hidden_layer_width],stddev=0.1),name='weights')

	with tf.name_scope('biais'):
		b1= tf.Variable(tf.truncated_normal([hidden_layer_width],stddev=0.1),name='weights')
	
	with tf.name_scope('activation'):
		y1 = tf.nn.sigmoid(tf.matmul(x,W1) + b1)	


with tf.name_scope('2nd_layer'):
	with tf.name_scope('weights'):
		W2 = tf.Variable(tf.truncated_normal([hidden_layer_width,10],stddev=0.1), name='weights')
	
	with tf.name_scope('biais'):
		b2= tf.Variable(tf.truncated_normal([10],stddev=0.1),name='weights')
		
	y =  tf.nn.softmax(tf.matmul(y1,W2) + b2)

#y = tf.nn.dropout(yf,0.95)

with tf.name_scope('cross_entropy'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), 
								   reduction_indices = [1]))
    
#en réalité si l'on veut réaliser ce calcul, on préfèrera utiliser
#la fct tf.nn.softmax_cross_entropy_with_logits car elle est plus efficace

#la valeur d'alpha a été choisie par dichotomie, c'est celle qui me donne les meilleurs résultats
alpha = 0.565

#la méthode minimize() combine les méthode compute_gradients() et apply_gradients()
#minimize() peut prendre bien plus de paramètres en entier, aller voir
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

#Comme une session normale, sauf qu'on peut intéragir avec dans le shell
#On peut lancer la session et faire les calculs après, à la volée
#Contrairement à la session normale, qui éxécute le graphe construit avant (à verifier)
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#Chaque tour de boucle, on récupère aléatoirement 100 données d'entrainement
#On fait le training avec une partie du training set : mini-batch gradient
#Avec un seul à chaque fois : stochatstic gradient


for actual_iter in range(number_of_training):
	
	percentage = actual_iter/number_of_training*100
	if (percentage%5==0) or (percentage==100):
		print("--",percentage,"% --")
        
	batch_xs, batch_ys = mnist.train.next_batch(training_set_size)
	#ici on fourni x et y_, si on regarde plus haut, ce sont les seules variables
	#dont on a besoin pour faire tourner le modèle
	#on calcule y avec x, W et b. Et on a besoin de y_ pour la cross_entropy 
    
    #feed dict permet de remplacer n'importe quel tenseur du graph, et non que les 
    #placeholder       
	sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

with tf.name_scope('accuracy'):
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("\nmodel's accuracy (alpha=", alpha, ") :")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

writer = tf.summary.FileWriter('./logs',sess.graph)
writer.add_graph(tf.get_default_graph())

#On remarquera qu'il y a plusieurs minima locaux avec des valeurs de poids différents 
#(à chaque lancement du programme)
#print(sess.run(b2))