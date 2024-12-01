# Medical Dataset Directory

Place your medical datasets in this directory. The following datasets are supported:

1. `medical_qa_dataset.json` - Medical Q&A format
2. `mimic_processed.json` - MIMIC-III processed format
3. `healthcaremagic_100k.json` - HealthCareMagic format

## Dataset Formats

### Medical Q&A Format
```json
{
    "question": "Patient presents with fever, cough, and fatigue for 5 days",
    "answer": "Based on the symptoms presented..."
}
```

### MIMIC-III Format
```json
{
    "clinical_history": "Patient history...",
    "symptoms": "Current symptoms...",
    "diagnosis": "Final diagnosis...",
    "clinical_reasoning": "Reasoning for diagnosis..."
}
```

### HealthCareMagic Format
```json
{
    "patient_complaint": "Description of symptoms...",
    "doctor_response": "Medical diagnosis and advice..."
}
```

## Data Privacy
Ensure all datasets are properly de-identified and comply with relevant healthcare data privacy regulations (e.g., HIPAA).