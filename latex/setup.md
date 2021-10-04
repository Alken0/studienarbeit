# Setup Latex environment

1. install MiKTeX: https://miktex.org/download
    - open MikTex + check for updates
    - install package cm-super if you encouter error "scalable font not found" (https://tex.stackexchange.com/questions/10706/pdftex-error-font-expansion-auto-expansion-is-only-possible-with-scalable)
2. install TexStudio: https://www.texstudio.org/
    - configure TexStudio: https://tex.stackexchange.com/questions/156205/how-to-compile-bibtex-with-texstudio
        - Options > Configure > Build > "Show Advanced Options" > Default Compiler: txs:///pdflatex | txs:///biber | txs:///pdflatex | txs:///view-pdf
        - make sure you use biber for compiling: https://tex.stackexchange.com/questions/154751/biblatex-with-biber-configuring-my-editor-to-avoid-undefined-citations/154754#154754
