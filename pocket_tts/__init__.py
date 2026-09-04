from pocket_tts.models.model_state import export_model_state
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.text_normalization import DictionaryEntry, UserDictionary

# Public methods:
# TTSModel.device
# TTSModel.sample_rate
# TTSModel.load_model
# TTSModel.generate_audio
# TTSModel.generate_audio_stream
# TTSModel.get_state_for_audio_prompt

__all__ = ["DictionaryEntry", "TTSModel", "UserDictionary", "export_model_state"]
