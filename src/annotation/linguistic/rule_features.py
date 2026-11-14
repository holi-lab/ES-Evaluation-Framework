"""Extract linguistic seeker features without LLM prompting."""


def extract_linguistic_features(turns, tokenizer_name, verbosity_bins, turn_bins):
    """Return verbosity, dialogue-length, and profanity features for one dialogue."""
    client_text = [text for speaker, text in turns if speaker == "Client"]
    if tokenizer_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in client_text]
    else:
        lengths = [len(text.split()) for text in client_text]

    average = sum(lengths) / max(1, len(lengths))
    b1, b2, b3, b4 = verbosity_bins
    verbosity = "very_short" if average < b1 else "short" if average < b2 else "medium" if average < b3 else "long" if average < b4 else "very_long"

    from profanity_check import predict
    profanity = bool(predict(client_text).sum()) if client_text else False

    short_max, medium_max = turn_bins
    total_turns = len(turns)
    turn_level = "long" if total_turns >= medium_max else "medium" if total_turns >= short_max else "short"
    return {
        "verbosity_level": verbosity,
        "total_dialogue_turns_level": turn_level,
        "user_profanity_flag": profanity,
    }
