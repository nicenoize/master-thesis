from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer

class EvaluationMetrics:
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        return wer(reference, hypothesis)

    @staticmethod
    def calculate_bleu(reference: List[str], hypothesis: str) -> float:
        return sentence_bleu([reference.split()], hypothesis.split())

    @staticmethod
    def calculate_latency(start_time: float, end_time: float) -> float:
        return end_time - start_time

    @staticmethod
    def calculate_throughput(input_size: int, processing_time: float) -> float:
        return input_size / processing_time

    @staticmethod
    def calculate_cost(api_calls: int, cost_per_call: float) -> float:
        return api_calls * cost_per_call

    @staticmethod
    def evaluate_model(results: Dict) -> Dict:
        evaluation = {}
        evaluation['wer'] = EvaluationMetrics.calculate_wer(results['reference'], results['transcription'])
        evaluation['bleu'] = EvaluationMetrics.calculate_bleu(results['reference_translation'], results['translation'])
        evaluation['latency'] = EvaluationMetrics.calculate_latency(results['start_time'], results['end_time'])
        evaluation['throughput'] = EvaluationMetrics.calculate_throughput(results['input_size'], evaluation['latency'])
        evaluation['cost'] = EvaluationMetrics.calculate_cost(results['api_calls'], results['cost_per_call'])
        return evaluation