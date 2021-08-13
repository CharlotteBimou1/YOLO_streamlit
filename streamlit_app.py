##### Import des librairies #####
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

############ 1- Définition de la fonction de détection d'objet ####################
def obj_detection(my_img):
    st.set_option('deprecation.showPyplotGlobalUse', False) #streamlit.set_option définit les options de configuration
#streamlit.pyplot() nécessite désormais la fourniture d'un chiffre
#Définir deprecation.showPyplotGlobalUse sur False désactivera l'avertissement de dépréciation
    column1, column2 = st.columns(2) #streamlit.beta_columns() insère des conteneurs disposés en colonnes côte à côte

    column1.subheader("Input image")#Affichage d'un sous-titre au-dessus de l'image d'entrée
    st.text("")
    ##Affichage de l'image d'entrée à l'aide de matplotlib
    plt.figure(figsize=(16, 16))
    plt.imshow(my_img)
    column1.pyplot(use_column_width=True)
    ## C://Users//lenoa//YOLO//Streamlit_yolov3//yolov3.weights : fichier des poids
    ## C://Users//lenoa//YOLO//Streamlit_yolov3//yolov3.cfg : fichier de configuration
######################## 2- Instanciation du modèle YOLO ############################
    # Import du fichier de configuration yolov3.cfg et du fichier de poids du modèle YOLOv3 yolov3.weights
    net = cv2.dnn.readNet("yolov3.weights",
                          "yolov3.cfg")

    labels = []
    # Import du fichier coco.names contenant les étiquettes de sortie de l'ensemble de données. Elles sont stockées dans une liste appelée labels
    with open("coco.names", "r") as f:
        labels = [line.strip() for line in f.readlines()] #La méthode strip() supprime les espaces de début et de fin des chaînes d'étiquettes
    names_of_layer = net.getLayerNames()
    output_layers = [names_of_layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(labels), 3))

    # load des images
    newImage = np.array(my_img.convert('RGB'))##Conversion de l'image en RVB
    img = cv2.cvtColor(newImage, 1)
    height, width, channels = img.shape ###Stockage de la hauteur, la largeur et le nombre de canaux de couleur de l'image

############## 3- Conversion des images en blobs en utilisant blobFromImage() ###############################
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    classID = []
    confidences = []
    boxes = []


    #Affichage des informations contenues dans la variable 'outputs' dans l'application
    for op in outputs:
        for detection in op:
            scores = detection[5:]#score de confiance
            class_id = np.argmax(scores) ##sélection de la classe de sortie ayant la probabilité maximale (score de confiance)
            confidence = scores[class_id]##Mise à jour de la variable 'confiance' avec le score de l'étiquette de sortie sélectionnée ci-dessus.
            if confidence > 0.5: #Si le score de confiance dépasse 0,5 (probabilité >50%), cela signifie qu'un objet a été détecté.
                                 # Il faut chercher alors à obtenir ses dimensions : center, width, height
                # Obtention de dimensions de l'object détecté : center,width,height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height) #centre de l'object
                w = int(detection[2] * width)  # width est la largeur originale de l'image
                h = int(detection[3] * height)  # height est la hauteur originale de l'image

                # Calcule des coordonnées de la boîte de délimitation (bounding box)
                x = int(center_x - w / 2)  ##Coordonnée x du coin supérieur gauche de la boîte
                y = int(center_y - h / 2)  ##Coordonnée Y du coin supérieur gauche de la boîte

                # Organisation des objets dans un tableau afin de pouvoir les extraire plus tard
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classID.append(class_id)
######################### 4- Ajustement du seuil de confiance et le seuil de NMS (Non-Maximum Suppression) #############
    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00, 1.00, 0.5, 0.01) #streamlit.slider() insère un widget de type slider
                                                                                       #Ses paramètres sont (étiquette textuelle affichée, valeur min, valeur max, intervalle de pas)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01) ##NMSBoxes() effectue le NMS en fonction des cases et des scores correspondants

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    print(indexes)

################### 5- Tracé des boîtes de délimitation (suite de la détection d'objet()) #################"
    items = [] ##Stockage des étiquettes du ou des objets détectés
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i] ##obtiention des dimensions de la ième boîte de délimitation à former
            ##obtiention du nom de l'objet détecté
            label = str.upper((labels[classID[i]]))
            color = colors[i] ##Couleur de la ième boîte de délimitation
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3) #Forme une boîte rectangulaire dont les paramètres sont (image, point de départ, point d'arrivée, couleur, épaisseur).
            items.append(label) ##Ajouter l'étiquette de sortie de l'objet borné
####################### 6- Afficher les objets détectés avec des boîtes d'ancrage (suite de object_detection())##############
    st.text("") ##Texte préformaté et à largeur fixe
    column2.subheader("Output image") ##Titre en haut de l'image de sortie
    st.text("")
    plt.figure(figsize=(15, 15)) #Tracez l'image de sortie avec les objets détectés en utilisant matplotlib
    plt.imshow(img)
    column2.pyplot(use_column_width=True)

    if len(indexes) > 1:
        # Texte à imprimer si l'image de sortie contient plusieurs objets détectés
        st.success("Found {} Objects - {}".format(len(indexes), [item for item in set(items)]))
    else:
        # Texte à imprimer si l'image de sortie a un seul objet détecté
        st.success("Found {} Object - {}".format(len(indexes), [item for item in set(items)]))


def main():
    st.title("Welcome to Object Detection using YOLOV3 Model")
    st.write(
        "Real-time object detection using YOLO model here. You can select one of the following options to proceed: ")
    st.write("- The 'See an illustration' option allows you to test the application with the default image of our template")
    st.write("- The 'Choose an image of your choice' option allows you to test the application with your own image using drag & drop")

    choice = st.radio("", ("See an illustration", "Choose an image of your choice"))
    # st.write()

    if choice == "Choose an image of your choice":#option 1: laissé la possibilité à l'utilisateur de choisir son image 'dentrée
        image_file = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            my_img = Image.open(image_file)
            obj_detection(my_img)

    elif choice == "See an illustration":#option 2: image par défaut teté dans notre modèle
        my_img = Image.open("img1.jpg")
        obj_detection(my_img)


if __name__ == '__main__':
    main()

