import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #mettre a '0' pour activer les warnings

import tensorflow as tf

print("\n")

node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0)

print("print des noeuds sans les exécuter")
print(node1, node2)

print("print des noeuds exécutés")
sess =  tf.Session()
print(sess.run([node1, node2]))

print("\n print du resultat des noeuds avec l'opération add")
#from __future__ import print_function
node3 = tf.add(node1,node2)
print("node3:", node3)
print("sess.run(node3):",sess.run(node3))

#permet de créer des variables a et b
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
print("\n print de a+b ou a et b sont des placeholders")
adder_node = a+b
print(sess.run(adder_node, {a : 3, b : 4.5} ))
print(sess.run(adder_node, {a: [3 , 4], b: [4.5 , 3]} ))

print("\n print de adder_node*3")
add_and_triple = adder_node*3
print(sess.run(add_and_triple, {a:3, b:4}))

print("\n")

#W et b sont des paramètres qui pourront être modifiés au cours du programme
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
#x voit sa valeur fixée (lors de l'appel de sess.run(..., {x: })) et ne change plus
x = tf.placeholder(tf.float32)
linear_model = W*x+b

#on créé un handler qui contient l'initialisation des variables
#et on le run dans notre session "sess" pour les initialiser
init = tf.global_variables_initializer()
sess.run(init)
print("resultats de W*x+b pour x =[1,2,3,4] et W=0.3, b=-0.3")
print(sess.run(linear_model, {x:[1,2,3,4]}))