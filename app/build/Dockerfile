FROM toshas/point2cad:v1


RUN python3 -m pip install --upgrade pip && \
    pip3 install gradio

WORKDIR ${ROOT_PROJECT}
EXPOSE 7860
WORKDIR ${ROOT_POINT2CAD}
