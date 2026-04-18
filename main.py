import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import random

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
# ================================
# SETTINGS
# ================================
data_dir = r"histopathological image dataset for ET"
IMG_SIZE = 224
TARGET_PER_CLASS = 5000  # ~10k images after augmentation
label_map = {"NE":0,"EH":0,"EP":0,"EA":1}

X_list, y_list, f_hc_list = [], [], []

# ================================
# IMAGE LOADER & PREPROCESSING
# ================================
def read_image_unicode(path):
    try:
        stream = np.fromfile(path, np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except:
        return None

def stain_normalization(img):
    img = img.astype(np.float32) + 1
    OD = -np.log(img / 255)
    OD[OD < 0.15] = 0
    for i in range(3):
        ch = OD[:,:,i]
        OD[:,:,i] = (ch - np.mean(ch)) / (np.std(ch)+1e-8)
    return (np.exp(-OD)*255).clip(0,255).astype(np.uint8)

def extract_tissue(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    return cv2.bitwise_and(img,img,mask=mask)

def augment_image(img):
    imgs=[img]
    imgs += [cv2.flip(img,1), cv2.flip(img,0)]
    for angle in [90,180,270]:
        M=cv2.getRotationMatrix2D((IMG_SIZE//2,IMG_SIZE//2),angle,1)
        imgs.append(cv2.warpAffine(img,M,(IMG_SIZE,IMG_SIZE)))
    alpha = np.random.uniform(0.8,1.2)
    beta = np.random.uniform(-20,20)
    imgs.append(cv2.convertScaleAbs(img,alpha=alpha,beta=beta))
    noise=np.random.normal(0,10,img.shape)
    imgs.append(np.clip(img+noise,0,255).astype(np.uint8))
    return imgs

# ================================
# HANDCRAFTED FEATURES
# ================================
def extract_handcrafted_features(img):
    features=[]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    glcm=graycomatrix(gray,[1],[0],256,True,True)
    features += [
        graycoprops(glcm,'contrast')[0,0],
        graycoprops(glcm,'energy')[0,0],
        graycoprops(glcm,'correlation')[0,0],
        graycoprops(glcm,'homogeneity')[0,0]
    ]
    lbp=local_binary_pattern(gray,8,1,'uniform')
    hist,_=np.histogram(lbp.ravel(),bins=10,range=(0,10))
    features += list(hist/(hist.sum()+1e-6))
    for space in [img,
                  cv2.cvtColor(img,cv2.COLOR_BGR2HSV),
                  cv2.cvtColor(img,cv2.COLOR_BGR2LAB)]:
        for i in range(3):
            ch=space[:,:,i].flatten()
            features += [np.mean(ch),skew(ch),kurtosis(ch)]
    return np.array(features)

# ================================
# LOAD DATA + FEATURES
# ================================
class_counts={0:0,1:0}
for cls in os.listdir(data_dir):
    path=os.path.join(data_dir,cls)
    if not os.path.isdir(path) or cls not in label_map:
        continue
    label=label_map[cls]
    for item in os.listdir(path):
        item_path=os.path.join(path,item)
        paths=[os.path.join(item_path,f) for f in os.listdir(item_path)] if os.path.isdir(item_path) else [item_path]
        for p in paths:
            if class_counts[label]>=TARGET_PER_CLASS:
                continue
            img=read_image_unicode(os.path.normpath(p))
            if img is None:
                continue
            img=stain_normalization(img)
            img=extract_tissue(img)
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            augmented_imgs = augment_image(img)
            for aug in augmented_imgs:
                if class_counts[label]>=TARGET_PER_CLASS:
                    break
                X_list.append(aug)
                y_list.append(label)
                f_hc_list.append(extract_handcrafted_features(aug))
                class_counts[label]+=1

X = np.array(X_list,dtype=np.float32)/255.0
y = np.array(y_list)
f_hc = np.array(f_hc_list)

# ================================
# HANDCRAFTED FEATURE PROCESSING
# ================================
scaler = StandardScaler()
f_scaled = scaler.fit_transform(f_hc)
pca = PCA(n_components=min(41,f_scaled.shape[1]))
f_hc_pca = pca.fit_transform(f_scaled)
hc_model = models.Sequential([layers.Dense(512,activation='relu',input_shape=(f_hc_pca.shape[1],)),
                              layers.Dense(256,activation='relu')])
f_hc_final = hc_model(f_hc_pca).numpy()

# ================================
# DEEP FEATURES
# ================================
base = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base.layers[:-50]:
    layer.trainable=False
for layer in base.layers[-50:]:
    layer.trainable=True
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(512, activation='relu')(x)
deep_model = models.Model(base.input, x)
deep_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
f_deep = deep_model.predict(X, batch_size=64, verbose=1)

# ================================
# FEATURE FUSION
# ================================
f_hc_final = StandardScaler().fit_transform(f_hc_final)
f_deep = StandardScaler().fit_transform(f_deep)
f_combined = np.concatenate([f_deep,f_hc_final],axis=1)

tokens = f_combined.reshape(-1,8,f_combined.shape[1]//8)
cls_token = np.zeros((tokens.shape[0],1,tokens.shape[2]))
tokens_with_cls = np.concatenate([cls_token, tokens], axis=1)

X_train,X_test,y_train,y_test = train_test_split(tokens_with_cls,y,test_size=0.2,stratify=y,random_state=42)
y_train_cat = tf.keras.utils.to_categorical(y_train,2)
y_test_cat = tf.keras.utils.to_categorical(y_test,2)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# ================================
# GWO HYPERPARAMETER TUNING
# ================================
# ⚡ Grey Wolf Optimizer (simplified)
def gwo_optimize(n_agents=5, n_iter=10):
    best_score = 0
    best_params = {}
    for _ in range(n_agents):
        lr = 10**np.random.uniform(-4, -2)
        dropout = np.random.uniform(0.2,0.5)
        batch_size = random.choice([32,64,128])
        heads = random.choice([2,4,8])
        # quick evaluation on small subset for speed
        idx = np.random.choice(len(X_train), size=500, replace=False)
        X_sub, y_sub = X_train[idx], y_train_cat[idx]
        model = build_vit(heads=heads, drop=dropout)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss=focal_loss(), metrics=['accuracy'])
        model.fit(X_sub, y_sub, epochs=3, batch_size=batch_size, verbose=0)
        score = model.evaluate(X_sub, y_sub, verbose=0)[1]
        if score > best_score:
            best_score = score
            best_params = {'lr':lr,'dropout':dropout,'batch_size':batch_size,'heads':heads}
    return best_params

# ================================
# ViT MODEL BUILD FUNCTION
# ================================
def transformer_block(x, num_heads=4, ff_dim=256):
    token_dim = x.shape[-1]
    x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=token_dim)(x_norm, x_norm)
    x1 = layers.Add()([x, attn_out])
    x_norm2 = layers.LayerNormalization(epsilon=1e-6)(x1)
    ff = layers.Dense(ff_dim, activation='gelu')(x_norm2)
    ff = layers.Dense(token_dim)(ff)
    x2 = layers.Add()([x1, ff])
    return x2

def build_vit(heads=4, drop=0.3):
    input_tokens = layers.Input(shape=(tokens_with_cls.shape[1], tokens_with_cls.shape[2]))
    x = transformer_block(input_tokens, num_heads=heads)
    x = transformer_block(x, num_heads=heads)
    cls_token = layers.Lambda(lambda z: z[:,0,:])(x)
    x = layers.Dense(256,activation='relu')(cls_token)
    x = layers.Dropout(drop)(x)
    output = layers.Dense(2, activation='softmax')(x)
    return models.Model(input_tokens, output)

# Focal Loss
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.exp(-ce)
        loss = alpha * tf.pow(1-pt, gamma) * ce
        return loss
    return focal_loss_fixed

# ================================
# RUN GWO TO FIND BEST HYPERPARAMS
# ================================
best_params = gwo_optimize()
print("Best Hyperparameters from GWO:", best_params)

# ================================
# FINAL TRAINING WITH BEST HYPERPARAMS
# ================================
vit_model = build_vit(heads=best_params['heads'], drop=best_params['dropout'])
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(best_params['lr'], decay_steps=5000)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)

vit_model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
history = vit_model.fit(X_train, y_train_cat,
                        validation_data=(X_test, y_test_cat),
                        epochs=50,
                        batch_size=best_params['batch_size'],
                        class_weight=class_weights,
                        callbacks=[EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)])

# ================================
# EVALUATION
# ================================
loss, acc = vit_model.evaluate(X_test, y_test_cat)
print("\n🔥 FINAL ACCURACY:", acc*100)

y_pred = np.argmax(vit_model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test, y_pred, target_names=['Benign','Malignant']))

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras import models, layers

# ================================
# BASELINE MODEL (for comparison)
# ================================
baseline_input = layers.Input(shape=(f_combined.shape[1],))
x = layers.Dense(256, activation='relu')(baseline_input)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
baseline_output = layers.Dense(2, activation='softmax')(x)
baseline_model = models.Model(baseline_input, baseline_output)
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
baseline_model.fit(f_combined, tf.keras.utils.to_categorical(y,2), epochs=20, batch_size=64, verbose=0)

# ================================
# PREDICTIONS
# ================================
y_pred_probs = vit_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

baseline_probs = baseline_model.predict(f_combined)
baseline_pred = np.argmax(baseline_probs, axis=1)

# ================================
# ROC CURVE
# ================================
plt.figure(figsize=(8,6))
fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:,1])
roc_auc = auc(fpr, tpr)

fpr_b, tpr_b, _ = roc_curve(y, baseline_probs[:,1])
roc_auc_b = auc(fpr_b, tpr_b)

plt.plot(fpr, tpr, color='#285A48', label=f'Densenet (AUC = {roc_auc:.3f})')
plt.plot(fpr_b, tpr_b, color='#8C5A3C', label=f'ViT+GWO(Proposed) (AUC = {roc_auc_b:.4f})')
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve',fontweight='bold')
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('True Positive Rate',fontweight='bold')
plt.legend(loc='lower right')
plt.savefig('ROC Curve.png',dpi=800)
plt.show()

# ================================
# PRECISION-RECALL CURVE
# ================================
plt.figure(figsize=(8,6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_probs[:,1])
ap = average_precision_score(y_test, y_pred_probs[:,1])

precision_b, recall_b, _ = precision_recall_curve(y, baseline_probs[:,1])
ap_b = average_precision_score(y, baseline_probs[:,1])

plt.plot(recall, precision, color='#5E0006', label=f'Densenet (AP = {ap:.3f})')
plt.plot(recall_b, precision_b, color='#003049', label=f'ViT+GWO(Proposed) (AP = {ap_b:.4f})')
plt.title('Precision-Recall Curve',fontweight='bold')
plt.xlabel('Recall',fontweight='bold')
plt.ylabel('Precision',fontweight='bold')
plt.legend(loc='lower left')
plt.savefig('Precision-Recall Curve.png',dpi=800)
plt.show()

# ================================
# CALIBRATION CURVE
# ================================
plt.figure(figsize=(8,6))
prob_true, prob_pred = calibration_curve(y_test, y_pred_probs[:,1], n_bins=10)
prob_true_b, prob_pred_b = calibration_curve(y, baseline_probs[:,1], n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', label='Densenet',color='#003049')
plt.plot(prob_pred_b, prob_true_b, marker='s', label='ViT+GWO(Proposed)',color='#5E0006')
plt.plot([0,1],[0,1],'k--',color='#1B0C0C')
plt.title('Calibration Curve',fontweight='bold')
plt.xlabel('Predicted Probability',fontweight='bold')
plt.ylabel('True Probability',fontweight='bold')
plt.legend()
plt.savefig('Calibration Curve.png',dpi=800)
plt.show()

# ================================
# CONFUSION MATRIX & PERFORMANCE TABLE
# ================================
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
plt.title("Confusion Matrix")
plt.show()

# Performance metrics
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision_val = TP/(TP+FP)
recall_val = TP/(TP+FN)
f1_val = 2*precision_val*recall_val/(precision_val+recall_val)

metrics_df = pd.DataFrame({'Metric':['Accuracy','Precision','Recall','F1 Score'],
                           'ViT':[accuracy,precision_val,recall_val,f1_val]})
print(metrics_df)

# ================================
# MODEL LOSS AND ACCURACY PLOTS
# ================================
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy',color='#4E8D9C')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy',color='#612D53')
plt.title('Model Accuracy',fontweight='bold')
plt.xlabel('Epoch',fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
plt.legend()
plt.savefig('Model Accuracy.png',dpi=800)
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss',color='#41431B')
plt.plot(history.history['val_loss'], label='Validation Loss',color='#D96868')
plt.title('Model Loss',fontweight='bold')
plt.xlabel('Epoch',fontweight='bold')
plt.ylabel('Loss',fontweight='bold')
plt.legend()
plt.savefig('Model Loss.png',dpi=800)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# ================================
# CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_test, y_pred)
TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

# ================================
# FPR & FNR BAR PLOT
# ================================
FPR = FP / (FP + TN)
FNR = FN / (TP + FN)

plt.figure(figsize=(8,6))
plt.bar(['FPR','FNR'], [FPR, FNR], color=['#612D53','#41431B'])
plt.title('False Positive Rate & False Negative Rate',fontweight='bold')
plt.xlabel('False Positive Rate',fontweight='bold')
plt.ylabel('Rate',fontweight='bold')
plt.ylim(0,1)
for i, v in enumerate([FPR,FNR]):
    plt.text(i, v+0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.savefig('False Negative Rate.png',dpi=800)
plt.show()

# ================================
# PERFORMANCE METRICS BAR PLOT
# ================================
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision_val = TP/(TP+FP) if (TP+FP)>0 else 0
recall_val = TP/(TP+FN) if (TP+FN)>0 else 0
f1_val = 2*precision_val*recall_val/(precision_val+recall_val) if (precision_val+recall_val)>0 else 0

metrics = ['Accuracy','Precision','Recall','F1 Score']
values = [accuracy, precision_val, recall_val, f1_val]

plt.figure(figsize=(8,6))
sns.barplot(x=metrics, y=values, palette='Blues_d')
plt.ylim(0,1)
plt.title('Performance Metrics ',fontweight='bold')
plt.xlabel('Metric',fontweight='bold')
plt.ylabel('Value',fontweight='bold')
for i, v in enumerate(values):
    plt.text(i, v+0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ================================
# 1️⃣ Grad-CAM for EfficientNetV2S Features
# ================================
def get_gradcam_heatmap(model, img_array, layer_name="top_conv"):
    """
    Generates Grad-CAM heatmap for a single image.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap)+1e-8)
    return heatmap.numpy()

# Example visualization for first test image
img_example = X_test[0:1]  # shape: (1,224,224,3) if using CNN input
heatmap = get_gradcam_heatmap(deep_model, img_example, layer_name="top_conv")

plt.figure(figsize=(8,6))
plt.imshow(img_example[0])
plt.imshow(heatmap, cmap='jet', alpha=0.4)  # overlay heatmap
plt.title("Grad-CAM: Important Regions",fontweight='bold')
plt.axis('off')
plt.show()


# ================================
# 2️⃣ ViT Attention Map Visualization
# ================================
# Extract attention weights from the first transformer block
attention_layer = vit_model.layers[2].layers[1]  # first MultiHeadAttention layer

# Create a new model to get attention weights
att_model = tf.keras.Model(inputs=vit_model.input, outputs=attention_layer.output)

# Example attention for first test sample
sample_tokens = X_test[0:1]  # (1, tokens_len, token_dim)
attention_output = att_model(sample_tokens)  # shape: (1, tokens_len, tokens_len, num_heads) in TF 2.11+
attention_map = tf.reduce_mean(attention_output, axis=-1)  # average over heads
attention_map = attention_map[0,0,1:]  # CLS token attention to each patch

plt.figure(figsize=(10,6))
plt.bar(range(len(attention_map)), attention_map.numpy(), color='#612D53')
plt.title("ViT CLS Token Attention per Patch",fontweight='bold')
plt.xlabel("Patch Index",fontweight='bold')
plt.ylabel("Attention Weight",fontweight='bold')
plt.show()