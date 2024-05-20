from fastapi import FastAPI, Request, UploadFile, HTTPException, status
import predictor
import image_downloader
import logging

from fastapi.responses import HTMLResponse
import aiofiles

import os

downloadsPath = r"D:\Projects\BirdClassification\downloads"

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename='logging.log', level=logging.DEBUG)
logging.getLogger("multipart").setLevel(logging.ERROR)

app = FastAPI()



@app.get("/urlpredict/{image_url:path}")
async def root(image_url:str):
	try:
		imagePath = image_downloader.download_image(image_url)
		prediction = predictor.predict(imagePath)
		logging.info('prediction {} {}'.format(imagePath, prediction)) 
		return {'result':prediction}
	except:
		logger.error('image_not_downloaded {}'.format(image_url))
		return {'error':'image not downloaded'}


@app.post('/upload')
async def upload(file: UploadFile):
	try:
		contents = await file.read()
		imagePath = os.path.join(downloadsPath, file.filename)
		print(imagePath)
		async with aiofiles.open(imagePath, 'wb') as f:
		    await f.write(contents)
		prediction = predictor.predict(imagePath)
		logging.info('prediction {} {}'.format(imagePath, prediction)) 
		return {'result':prediction}
		
	except Exception:
		logger.error('image_not_uploaded')
		return {'error':'There was an error uploading the file'}

		# raise HTTPException(
			# status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			# detail='There was an error uploading the file',
		# )
	finally:
		await file.close()



@app.get('/filepredict')
async def main():
	content = '''
	<body>
	<form action='/upload' enctype='multipart/form-data' method='post'>
	<input name='file' type='file'>
	<input type='submit'>
	</form>
	</body>
	'''
	return HTMLResponse(content=content)


if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=5049)