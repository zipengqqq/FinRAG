from starlette.responses import JSONResponse
from fastapi.encoders import jsonable_encoder


def build_response(data) -> JSONResponse:
    encoded = jsonable_encoder(data)
    return JSONResponse(status_code=200, content={'data': encoded, 'message': 'success'})

def success_response():
    return JSONResponse(status_code=200, content={'message': 'success'})
