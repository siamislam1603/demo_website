/**
 * @returns current time as number
 */
function getName() {
  return +new Date();
}

const STREAM_NAME = getName();

/**
 * @returns if the browser has video devices
 */
function permittedGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

/**
 * @description sends video chunks to server
 * @param {Blob} file
 * @param {number} chunkNumber
 */
function sendFile(file, chunkNumber) {
  const serverURL = "http://127.0.0.1:8000";
  const uploadMethod = "POST";

//  {'data': file};

  const formData = new FormData();
  formData.append("file", file);
  formData.append("name", STREAM_NAME);
  formData.append("chunk", chunkNumber);

  fetch(serverURL, {
    method: uploadMethod,
    body: formData,
//    headers : {
//    'Content-Type': 'multipart/form-data',
//    },
  });
}

/**
 * @description takes stream from device then uploads to server and shows in video source element
 * @param {MediaStream} stream
 * @param {MediaSource} mediaSource
 */
function processStream(stream, mediaSource) {
  const videoBuffer = mediaSource.addSourceBuffer("video/webm;codecs=vp8");
  const mediaRecorder = new MediaRecorder(stream);
  let countUploadChunk = 0;

  mediaRecorder.ondataavailable = (data) => {
    sendFile(data.data, countUploadChunk);

    data.data.arrayBuffer().then((res) => {
      videoBuffer.appendBuffer(res);
    });

    countUploadChunk++;
  };

  mediaRecorder.start();

  setInterval(() => {
    mediaRecorder.requestData();
  }, 1000);
}

/**
 * @description main block
 */
if (permittedGetUserMedia()) {
  const video = document.querySelector("video");
  const mediaSource = new MediaSource();
  video.src = URL.createObjectURL(mediaSource);

  navigator.mediaDevices
    .getUserMedia({
      video: true,
    })
    .then((stream) => processStream(stream, mediaSource));
}
