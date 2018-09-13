require('./faceLandmarks/.env')

const express = require('express')
const path = require('path')
const fs = require('fs')

const app = express()

const publicDir = path.join(__dirname, './faceLandmarks')
app.use(express.static(publicDir))
app.use(express.static(path.join(__dirname, './shared')))
app.use(express.static(path.join(__dirname, './node_modules/file-saver')))
app.use(express.static(path.join(__dirname, '../../examples/public')))
app.use(express.static(path.join(__dirname, '../../weights')))
app.use(express.static(path.join(__dirname, '../../dist')))

const trainDataPath = path.resolve(process.env.TRAIN_DATA_PATH)
app.use(express.static(trainDataPath))

const pngPath = path.join(trainDataPath, 'png')
const jpgPath = path.join(trainDataPath, 'jpg')
const groundTruthPath = path.join(trainDataPath, 'pts')
app.use(express.static(pngPath))
app.use(express.static(jpgPath))
app.use(express.static(groundTruthPath))

const trainFilenames = JSON.parse(fs.readFileSync(path.join(publicDir, './trainData.json')))
const trainFilenamesSet = new Set(trainFilenames)
const testFilenames = fs.readdirSync(groundTruthPath).filter(file => !trainFilenamesSet.has(file))
app.get('/face_landmarks_train_filenames', (req, res) => res.status(202).send(trainFilenames))
app.get('/face_landmarks_test_filenames', (req, res) => res.status(202).send(testFilenames))

app.get('/', (req, res) => res.redirect('/train'))
app.get('/train', (req, res) => res.sendFile(path.join(publicDir, 'train.html')))
app.get('/verify', (req, res) => res.sendFile(path.join(publicDir, 'verify.html')))

app.listen(8000, () => console.log('Listening on port 8000!'))