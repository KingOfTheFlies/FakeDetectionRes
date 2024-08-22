#!/bin/bash

# Директория, в которой нужно переименовать файлы
DIRECTORY="./UniversalFakeDetect/guided/0_real"

# Переход в указанную директорию
cd "$DIRECTORY" || exit

# Перебор всех файлов с расширением .JPEG и переименование их в .jpg
for FILE in *.JPEG; do
    # Проверка, что файл существует
    if [ -e "$FILE" ]; then
        # Получение имени файла без расширения
        BASENAME=$(basename "$FILE" .JPEG)
        # Переименование файла
        mv "$FILE" "$BASENAME.jpg"
    fi
done

echo "Переименование завершено."