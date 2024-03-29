from tensorflow.keras import preprocessing as preproc
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential as seq, load_model as load
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.metrics import Accuracy, AUC, Precision, Recall, SpecificityAtSensitivity
from pandas import DataFrame as df

datagen = preproc.image.ImageDataGenerator(
    validation_split=.18,
    rescale=1./255,
    brightness_range=[25.5, 65.5],
    #shear_range=0.3,
    zoom_range=0.2,
    #horizontal_flip=True,
)

train = datagen.flow_from_directory(
    directory="data/plot_epi/train/",
    class_mode="categorical",
    color_mode="rgb",
    target_size=(299, 299),
    shuffle=True,
    interpolation="bilinear",
    seed=42,
    subset="training",
)

val = datagen.flow_from_directory(
    directory="data/plot_epi/train/",
    class_mode="categorical",
    color_mode="rgb",
    target_size=(299, 299),
    shuffle=False,
    interpolation="bilinear",
    seed=42,
    subset="validation",
)

epi_InceptionResNetV2_model = seq([
    Input(shape=(299, 299, 3)),
    InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(299, 299, 3)),
    GlobalAveragePooling2D(),
    Dense(2, activation="softmax", kernel_regularizer="l1_l2")
], name="EPI_InceptionResNetV2")

epi_InceptionResNetV2_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adadelta(learning_rate=1e-2), 
    metrics=["acc", Precision(.51), Recall(.51), SpecificityAtSensitivity(.5), AUC()]
)

callbacks = [
    ModelCheckpoint(filepath="ckpt/checkpoint-augment-l1-l2-inceptionresnetv2-epi-{epoch:02d}-{val_acc:.3f}.h5", monitor="val_acc", save_best_only=True, mode="max"),
    TerminateOnNaN()
]

epi_InceptionResNetV2_model_result = epi_InceptionResNetV2_model.fit(
    x=train, validation_data=val, epochs=30, callbacks=callbacks)

epi_InceptionResNetV2_model.save("model/augment_l1_l2_epi_InceptionResNetV2_model.h5")
epi_InceptionResNetV2_model.save_weights("model/augment_l1_l2_epi_InceptionResNetV2_weights.h5")
df.from_dict(epi_InceptionResNetV2_model_result.history).to_csv('result/augment_l1_l2_epi_InceptionResNetV2_model_result.csv', index=False)