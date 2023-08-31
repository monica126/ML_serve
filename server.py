# This file is entry point for our REST API service
@app.get('/health')
def health_check():
    content = {'Server status': 'Ok', 'DB connection': 'Ok'}

    # check DB connection and get the latest ML pipeline version
    try:
        ml_pipeline_version = get_latest_ml_pipeline_version()
    except Exception:
        content['DB connection'] = 'DB unavailable'

    # check if the ML pipeline exists
    if os.path.isfile(os.path.join(ML_PIPELINES_PATH, f'pipeline_{ml_pipeline_version}.pickle')):
        content['ML pipeline'] = ml_pipeline_version
    else:
        content['ML pipeline'] = 'ML pipeline unavailable'

    return JSONResponse(content=content)
  
########### Prediction w.o using model caching#################
@app.post('/predict')
def predict(item: Item, response: Response):
    # load the ML pipeline
    try:
        ml_pipeline = load(os.path.join(ML_PIPELINES_PATH, 'ml_pipeline.pickle'))
    except FileNotFoundError:
        logger.error(f'The ML pipeline does not exist.')

        response.status_code = status.HTTP_404_NOT_FOUND
        return JSONResponse(content={'Status': 'The model was not found.'},
                            status_code=status.HTTP_404_NOT_FOUND)

    # make prediction
    df = convert_item_to_df(item)
    prediction = ml_pipeline.predict(df)

    return JSONResponse(content={'prediction': prediction})
########### Prediction w. model caching ##############

CACHE: Dict[str, Any] = {}


@app.post('/predict')
async def predict(item: Item, response: Response):
    pipeline_version = get_latest_ml_pipeline_version()

    # check if we have the latest ml pipeline in the cache. Otherwise, load and cache the newest.
    if CACHE.get('ml_pipeline_version', None) != pipeline_version:
        try:
            ml_pipeline = load(os.path.join(ML_PIPELINES_PATH, f'pipeline_{pipeline_version}.pickle'))

            # cache new ml pipeline and ml pipeline version.
            CACHE['ml_pipeline'] = ml_pipeline
            CACHE['ml_pipeline_version'] = pipeline_version
        except FileNotFoundError:
            logger.error(f'The ML pipeline version "{pipeline_version}" does not exist.')

            response.status_code = status.HTTP_404_NOT_FOUND
            return JSONResponse(content={'Status': 'The model was not found.'},
                                status_code=status.HTTP_404_NOT_FOUND)

    # make prediction
    df = convert_item_to_df(item)
    prediction = CACHE['ml_pipeline'].predict(df)[0]

    # save features + prediction + pipeline version
    await save_prediction(df.values.tolist()[0] + [prediction, pipeline_version])

    return JSONResponse(content={'prediction': prediction})
