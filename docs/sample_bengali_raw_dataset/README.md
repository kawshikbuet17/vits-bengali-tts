# Sample Bengali Raw Dataset Layout

This folder is a tiny structural example for `scripts/prepare_bengali_dataset.py`.

It is **not** a runnable training dataset because it does not include a valid `.flac` audio file. The real remote dataset must contain matching `.flac` and `.json` files for each utterance.

Expected real layout:

```text
TTS_Dataset/
  Male/
    01332512906/
      2024-05-07/
        00d4da35-bebc-471b-aa92-0d9398388e98.flac
        00d4da35-bebc-471b-aa92-0d9398388e98.json
    01700000001/
      male-no-date-0001.flac
      male-no-date-0001.json
  Female/
    01800000001/
      2024-05-08/
        female-date-0001.flac
        female-date-0001.json
    01900000001/
      female-no-date-0001.flac
      female-no-date-0001.json
```

This sample intentionally includes:

- 2 male speaker folders
- 2 female speaker folders
- one example with a date folder
- one example without a date folder

The sample JSON in this folder shows the metadata shape and transcript location:

```text
annotation[*]["sentence"]
```

Do not use this folder for training unless you replace the placeholder with a real `.flac` file.

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  
