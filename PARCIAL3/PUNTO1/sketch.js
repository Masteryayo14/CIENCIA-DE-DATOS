let model;
let video;
let label = "Cargando modelo...";

function preload() {
  // Carga el modelo exportado desde Teachable Machine
  model = ml5.imageClassifier('modelo/model.json', videoReady);
}

function setup() {
  createCanvas(640, 480);
  
  // Captura de video
  video = createCapture(VIDEO);
  video.hide();
  
  // Comienza la clasificación del video
  classifyVideo();
}

function videoReady() {
  console.log("¡El video está listo!");
}

function classifyVideo() {
  // Clasifica la imagen del video
  model.classify(video, gotResult);
}

function gotResult(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  
  // Actualiza la etiqueta con el resultado
  label = results[0].label;
  
  // Vuelve a clasificar
  classifyVideo();
}

function draw() {
  background(220);
  
  // Muestra el video
  image(video, 0, 0);
  
  // Muestra la etiqueta
  textSize(32);
  fill(255);
  text(label, 10, height - 10);
}
