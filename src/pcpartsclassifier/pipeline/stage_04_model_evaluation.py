from pcpartsclassifier.config.configuration import ConfigurationManager
from pcpartsclassifier.components.model_evaluation import ModelEvaluation
from pcpartsclassifier import logger

STAGE_NAME = 'Model Evaluation'

class ModelEvaluationPipeline:
  def __init__(self) -> None:
    pass

  def main(self):
    config = ConfigurationManager()
    evaluation_config = config.get_evaluation_config()
    evaluation = ModelEvaluation(config=evaluation_config)
    evaluation.evaluate()
    evaluation.save_score()
    evaluation.log_into_mlflow()

if __name__ == '__main__':
  try:
    logger.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<< \n x===================x')
  except Exception as e:
    logger.exception(e)
    raise e