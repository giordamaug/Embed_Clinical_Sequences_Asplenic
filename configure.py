from embedding import LSTMembedder, BEHRTembedder, COUNTEREmbedder, TimeAwareLSTMEmbedder, DipoleEmbedder 
from embedding import StaticEmbedder, RETAINembedder, DOMEEmbedder, BINARYEmbedder, GRUEmbedder, GRUEDembedder

def configure_embedder(event_sequences, event_sequences_no_trunc, visit_sequences, labels, targets, dataset,
                       word_to_idx, code2id, vocab, attributes, num_epochs, batch_size, embedding_dim, hidden_dim, enable_plot):
    return  { 
        "LSTM" : 
        {   "func": LSTMembedder,
            "kwargs": {
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": word_to_idx,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "embed_size": embedding_dim,
                "hidden_size": hidden_dim,
                "enable_plot": enable_plot
            }
        },
      "RETAIN" : 
        {   "func": RETAINembedder,
            "kwargs": {
                "sequences": visit_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "embed_size": embedding_dim,
                "hidden_size": hidden_dim,
                "enable_plot": enable_plot
            }
        },
      "BEHRT" : 
        {   "func": BEHRTembedder,
            "kwargs": {
                "sequences": visit_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "embed_size": embedding_dim,
                "hidden_size": hidden_dim,
                "enable_plot": enable_plot
            }
        },
      "Dipole" : 
        {   "func": DipoleEmbedder,
            "kwargs": {
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "embed_size": embedding_dim,
                "hidden_size": hidden_dim,
                "enable_plot": enable_plot
            }
        },
      "tLSTM" : 
        {   "func": TimeAwareLSTMEmbedder,
            "kwargs": {
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": word_to_idx,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "embed_size": embedding_dim,
                "hidden_size": hidden_dim,
                "enable_plot": enable_plot
            }
        },
       "GRU-D" : 
        {   "func": GRUEDembedder,
            "kwargs": {
                "sequences": visit_sequences,
                "labels": labels,
                "word_to_idx": code2id,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "embed_size": embedding_dim,
                "hidden_size": hidden_dim,
                "enable_plot": enable_plot
            }
        },
      "GRU" : 
        {   "func": GRUEmbedder,
            "kwargs": {
                "sequences": event_sequences,
                "labels": labels,
                "word_to_idx": word_to_idx,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "embed_size": embedding_dim,
                "hidden_size": hidden_dim,
                "enable_plot": enable_plot
            }
        },
      "STATIC" : 
        {   "func": StaticEmbedder,
            "kwargs": {
                "df": dataset,
                "include_attributes": attributes,
                "enable_plot": enable_plot
            }
        },
       "DOME" :
        {
            "func": DOMEEmbedder,
            "kwargs": {
                 "sequences": event_sequences_no_trunc,
                 "targets": targets,
                 "df": dataset,
                 "enable_plot": enable_plot
            }
        },
        "BINARY": 
        {
            "func": BINARYEmbedder,
            "kwargs": {
                "sequences": event_sequences,
                "vocab": vocab,
                "targets": targets,
                "enable_plot": enable_plot
            }
        },
        "COUNTER": 
        {
            "func": COUNTEREmbedder,
            "kwargs": {
                "sequences": event_sequences,
                "vocab": vocab,
                "targets": targets,
                "enable_plot": enable_plot
            }
        }
    }
