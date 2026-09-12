"""Gemeinsame Dataset-Logik fuer Train- und Test-Engine.

Beide Engines hatten eigene Kopien derselben Ordner-Suche — mit jeweils
anderen Luecken (train/ + test/ ohne val/, leere Split-Ordner, Parquet mit
Bildbytes). Die Regeln stehen jetzt an genau einer Stelle.
"""
