"""legalrag - Arabic Legal Q&A RAG (JEV-RAG) over the Egyptian Civil Code."""

import truststore

# use the OS certificate store for HTTPS: machines with TLS inspection (antivirus, corporate
# proxy) install their root certificate there, and certifi does not know it
truststore.inject_into_ssl()

__version__ = "0.1.0"
