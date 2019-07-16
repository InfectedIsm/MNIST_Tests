#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:24:16 2018

@author: infected
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np
import datetime
import progressbar

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True )	


# =============================================================================
# timer qui est un peu pourri comparé à la librarie progressbar
# =============================================================================
#def remaining_time(iteration, max_iter, flag_timer=0):
##	percentage = iteration/max_iter*100
#	if flag_timer == 0:
#		start = time.time()
#		flag_timer=1	
#	
#	if percentage == 5:
#		end = time.time()
#		time_in_mn = round((end - start)*100/5/60,2)
#		print("estimated remaining time",int(time_in_mn) ,"mn", int((time_in_mn-int(time_in_mn))*60),"s")
#		print("actual time :" , datetime.datetime.now().hour, ":", datetime.datetime.now().minute,":",datetime.datetime.now().second)
#    
#	if (percentage%5==0) or (percentage==100):
#		print("--",percentage,"% --")
#		
#		return flag_timer


number_of_training = 20000
training_set_size = 10

K=4     #largeur de la première couche conv (nombre de tranches)
L=8     #largeur de la seconde couche conv
M=12    #largeur de la troisieme couche conv

with tf.device('/gpu:0'):
    X = tf.placeholder(tf.float32,[None,28,28,1])
    
    #5 et 5 pour la taille du de la fenetre, 1 pour le nombre de couche de l'image en entrée (n&b 1, rgb 3)
    #K pour le nombre de tranches de la première couche
    #le nombre de tranche équivaut au nombre de neurones pour chacune des fenetres 
    W1 = tf.Variable(tf.truncated_normal([5,5,1,K], stddev=0.1))
    #ici ones([K]) genere un vecteur de taille K
    B1 = tf.Variable(tf.ones([K])/10)
    
    W2 = tf.Variable(tf.truncated_normal([5,5,K,L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L])/10)
    
    W3 = tf.Variable(tf.truncated_normal([4,4,L,M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M])/10)
    
    
    #Au dessus nous avions les couches convolutionnelles
    #Maintenant nous avons les couches denses
    N=200
    #Chacun des neurones de la couches denses (au nombre de N) est relié à tous les neurones de la couche
    #convolutionnelle supérieure (au nombre de 4*4*M)
    W4 = tf.Variable(tf.truncated_normal([7*7*M,N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N])/10)
    
    W5 = tf.Variable(tf.truncated_normal([N,10], stddev=0.1))
    B5 = tf.Variable(tf.ones([10])/10)
    
    #Ici, on choisi un stride de 1 (les deux chiffres du centre), pour les autres je ne sais pas
    #a chaque opération de convolution, la taille de Yc évolue
    #a Yc1, on avait une image 28*28*1, comme on a un stride de 1 avec un padding conservant l'image originale
    # et une couche W1 de profondeur K=4, on a donc Yc1 de taille 28*28*4 
    Yc1 = tf.nn.relu(tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding="SAME") + B1)
    #on passe, a cause du stride de 2 et de L a une talle de 14*14*8
    Yc2 = tf.nn.relu(tf.nn.conv2d(Yc1,W2,strides=[1,2,2,1], padding="SAME") + B2)
    #on passe, a cause du stride de 2 et de M a une talle de 7*7*12
    Yc3 = tf.nn.relu(tf.nn.conv2d(Yc2,W3,strides=[1,2,2,1], padding="SAME") + B3)
    
    #afin d'associer à chaque neurone de la couche dense les neurones de la couche conv du dessus
    #on l'applatie via la fonction reshape, en faisant un vecteur de taille 7*7*M
    #en realité, comme on feed notre réseau avec un batch de plusieurs images, on met au début -1
    #cela veut dire que l'on demande a tensorflow de s'arranger pour faire une matrice de la bonne taille
    #et il y a une seule solution pour ca
    YY = tf.reshape(Yc3, shape=[-1,7*7*M])
    
    #Ici on fait simplement la multiplication des poids de la couche dense avec les neurones de la couches supérieur
    #et on utilise comme fonction d'activation reLU
    Yd = tf.nn.relu(tf.matmul(YY,W4)+B4)
    Ydrop = tf.nn.dropout(Yd,0.75)
    #Et on fini avec un softmax pour obtenir des proba pour chaque nombre
    y = tf.nn.softmax(tf.matmul(Ydrop,W5)+B5)
    
    # =============================================================================
    # a partir d'ici, j'ai c/c le code d'entrainement du tuto 1
    # =============================================================================
    
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), 
    						reduction_indices = [1]))
        
    #en réalité si l'on veut réaliser ce calcul, on préfèrera utiliser
    #la fct tf.nn.softmax_cross_entropy_with_logits car elle est plus efficace
    
    #la valeur d'alpha a été choisie par dichotomie, c'est celle qui me donne les meilleurs résultats
    alpha = 0.003
    
    #la méthode minimize() combine les méthode compute_gradients() et apply_gradients()
    #minimize() peut prendre bien plus de paramètres en entier, aller voir
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)
    
#Comme une session normale, sauf qu'on peut intéragir avec dans le shell
#On peut lancer la session et faire les calculs après, à la volée
#Contrairement à la session normale, qui éxécute le graphe construit avant (à verifier)
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#Chaque tour de boucle, on récupère aléatoirement 100 données d'entrainement
#On fait le training avec une partie du training set : mini-batch gradient
#Avec un seul à chaque fois : stochatstic gradient

#flag_timer=0
with progressbar.ProgressBar(max_value=number_of_training) as bar:
	for actual_iter in range(number_of_training):
		batch_xs, batch_ys = mnist.train.next_batch(training_set_size)
		
		bar.update(actual_iter)
	
		
		#ici on fourni x et y_, si on regarde plus haut, ce sont les seules variables
		#dont on a besoin pour faire tourner le modèle
		#on calcule y avec x, W et b. Et on a besoin de y_ pour la cross_entropy 
	    
	    #feed dict permet de remplacer n'importe quel tenseur du graph, et non que les 
	    #placeholder       
		sess.run(train_step, feed_dict={X: np.reshape(batch_xs,[-1,28,28,1]), y_:batch_ys})
	
	
        
        
#On va maintenant tester le modèle
#Cette fonction renvoie True (booléen) si dans tf.equal(a,b) a et b sont égaux
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#Ensuite on caste les booléen en float32, puis on calcule la moyenne
accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("\nmodel's accuracy (alpha=", alpha, ") :")
print(sess.run(accuracy, feed_dict={X: np.reshape(mnist.test.images,[-1,28,28,1]), y_: mnist.test.labels}))

#On remarquera qu'il y a plusieurs minima locaux avec des valeurs de poids différents 
#(à chaque lancement du programme)
#print(sess.run(b2))



