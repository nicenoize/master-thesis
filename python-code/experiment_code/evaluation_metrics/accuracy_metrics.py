from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics import edit_distance

def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    distance = edit_distance(ref_words, hyp_words)
    return distance / len(ref_words)

def calculate_bleu(reference: List[str], hypothesis: str) -> float:
    return sentence_bleu([reference.split()], hypothesis.split())

def evaluate_transcription_accuracy(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    wer_scores = [calculate_wer(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return {
        "average_wer": sum(wer_scores) / len(wer_scores),
        "best_wer": min(wer_scores),
        "worst_wer": max(wer_scores)
    }

def evaluate_translation_accuracy(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    bleu_scores = [calculate_bleu(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return {
        "average_bleu": sum(bleu_scores) / len(bleu_scores),
        "best_bleu": max(bleu_scores),
        "worst_bleu": min(bleu_scores)
    }

def evaluate_sentiment_accuracy(true_labels: List[str], predicted_labels: List[str]) -> float:
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    return correct / len(true_labels)

def evaluate_accuracy(api_results: Dict, local_results: Dict) -> Dict[str, Dict[str, float]]:
    accuracy_results = {}
    
    for model_type in ["transcription", "translation", "sentiment"]:
        if model_type == "transcription":
            api_accuracy = evaluate_transcription_accuracy(api_results[f"{model_type}_references"], api_results[f"{model_type}_hypotheses"])
            local_accuracy = evaluate_transcription_accuracy(local_results[f"{model_type}_references"], local_results[f"{model_type}_hypotheses"])
        elif model_type == "translation":
            api_accuracy = evaluate_translation_accuracy(api_results[f"{model_type}_references"], api_results[f"{model_type}_hypotheses"])
            local_accuracy = evaluate_translation_accuracy(local_results[f"{model_type}_references"], local_results[f"{model_type}_hypotheses"])
        else:  # sentiment
            api_accuracy = evaluate_sentiment_accuracy(api_results[f"{model_type}_true_labels"], api_results[f"{model_type}_predicted_labels"])
            local_accuracy = evaluate_sentiment_accuracy(local_results[f"{model_type}_true_labels"], local_results[f"{model_type}_predicted_labels"])
        
        accuracy_results[model_type] = {
            "api": api_accuracy,
            "local": local_accuracy
        }
    
    return accuracy_results