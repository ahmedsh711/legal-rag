"""legalrag - Arabic Legal Q&A RAG (JEV-RAG) over the Egyptian Civil Code."""

import truststore

# Use the operating system's certificate store for every HTTPS call (Hugging Face, OpenRouter,
# Jev). Machines with HTTPS inspection (antivirus, corporate proxy) add their root certificate
# to the OS store, which Python's bundled certifi list does not know about.
truststore.inject_into_ssl()

__version__ = "0.1.0"
