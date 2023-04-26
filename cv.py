import cv2
import torch

from modelo import modelo

# Carregar o modelo treinado
model = modelo()
model.load_state_dict(torch.load('mnist_model.pt'))

# Definir uma função para pré-processar a imagem


def preprocess(image):
    # Converte para cinza a imagem
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Image Blurring (Image Smoothing), se usa para remover barulho
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshhold, serve para separar o background do foreground
    # cv2.adaptiveThreshold é muito lento, mas dá resultados melhores
    _, thresh = cv2.threshold(
        blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize para 28x28 a imagem
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

    # Transforma a imagem em um tensor
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)

    # Normaliza os pixels para que fiquem entre 0 e 1
    tensor /= 255.0

    return tensor


# Inicialize o objeto de captura de vídeo
cap = cv2.VideoCapture(0)


# Pega as medidas da imagem e define o tamanho do bounding box
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

bbox_size = (250, 250)
bbox = [(int(width // 2 - bbox_size[0] // 2), int(height // 2 - bbox_size[1] // 2)),
        (int(width // 2 + bbox_size[0] // 2), int(height // 2 + bbox_size[1] // 2))]

# Começa o vídeo
while True:
    _, frame = cap.read()

    # O que está no box é preprocessado e transformado em um tensor
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_tensor = preprocess(img_cropped)

    # Predição pelo modelo
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        confidence = torch.nn.functional.softmax(
            output, dim=1)[0][predicted] * 100

    # Desenha o bounding box apenas para visualização
    cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 3)

    # Definido um nível de confiança, o número é mostrado na tela
    if confidence > 23:
        cv2.putText(frame, str(predicted.item(
        )), (bbox[0][0] + 5, bbox[0][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Cria uma janela para mostrar o vídeo
    cv2.imshow('input', frame)

    # Cancela o loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) == ord('q'):
        break

# Sai do programa
cap.release()
cv2.destroyAllWindows()
