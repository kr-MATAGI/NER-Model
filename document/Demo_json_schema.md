## JSON Schema For Demo
  1. Request (Demo Web to API)

  ```json
    {
       "date": "yyyy-mm-dd_hh:mm:ss.msec",
       "sentences": [
        {"id": "sha256 or integer", "text": "str"}
       ]
    }
  ```
  
  2. Response (API to Demo)
  
  ```json
    {
      "date": "yyyy-mm-dd_hh:mm:ss.msec",
      "results": [{
        "id": "str", 
        "text": "str",
        "ne": [
          {"word": "str", "label": "str", "begin": "int", "end": "int", "length": "int"}
        ]
      }]
    }
  ```
