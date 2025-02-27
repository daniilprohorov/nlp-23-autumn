# Указания к выполнению лабораторных работ

В данном документе приведены общие рекомендации и требования по выполнению лабораторных работ, а также по оформлению полученных результатов.

## Выбор датасета

Перед выполнением лабораторных работ по данному курсу необходимо выбрать датасет, который будет использоваться в процессе работы. Так, на выбор предоставляется 4 варианта корпусов, отличающихся уровнем сложности:

1. Собственный датасет - вы можете найти и выбрать любой датасет в открытом доступе. Основными требованиями к датасету:
   1. Примерный объем датасета не менее 1 000 000 словоупотреблений.
   2. Если планируется выполнять в 4 лабораторной работе классификацию, необходимо иметь размеченные классы для фрагментов текста. Например, как в 3 варианте. Рекомендуемое количество классов 2-15.
   3. Желательно использовать английский язык из-за его простоты и количества моделей для работы с ним.
2. [Датасет, состоящий из сгенерированных текстов](https://bit.ly/generated-corpus) - наиболее простой с точки зрения автоматической обработки датасет, состоящий из автоматически сгенерированного текста, для которого гарантируется возможность построения токенизатора на основе регулярных выражений, обрабатывающего текст со 100%-ной точностью. Также гарантируется возможность разработки классификатора, обеспечивающего точность 100%;
3. [Базовый новостной датасет](https://huggingface.co/datasets/ag_news)(прямые ссылки на скачивание: [обучающая](https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv) и [тестовая](https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv) выборки) - датасет, предназначенный для решения задачи тематического моделирования и классификации текстов новостей. Данный корпус состоит из реальных текстов новостей и реализует сравнительно простую аннотационную схему классификации. Большинство текстов не сложны с точки зрения автоматической обработки, однако встречается ряд сложных случаев, что делает решение задач его обработки с идеальной точностью затруднительным;
4. [Усложненный новостной датасет](http://qwone.com/~jason/20Newsgroups/)(прямая [ссылка](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz) на скачивание) - датасет, задача обработки которого отличается наименьшей тривиальностью, что обусловлено сложностью аннотационной схемы, в которой предусмотрены иерархические зависимости между классами, а также высокой вариативностью структуры текстов относительно всех уровней разметки. Для получения высокой точности при обработке данного датасета может потребоватся более внимательное изучение его структуры и более основательный подход к выбору методов решения поставленной задачи.

Результат выбора датасета влияет на следующие аспекты процесса прохождения курса:
1. Глубина погружения в процесс решения практических задач обработки текстов на естественном языке;
1. Скорость выполнения лабораторных работ;
1. Количество и сложность вопросов, на которые потребуется ответить на защите;
1. Необходимость реализации функциональности, не предусмотренной заданием, для получения дополнительных баллов;
1. Возможность получения "автомата" за экзамен.

## Прочие рекомендации

Также перед выполнением лабораторных работ необходимо выбрать платформу для выполнения заданий курса (прежде всего, язык программирования). Для выбранной платформы должны быть доступны библиотеки, содержащие стандартную реализацию моделей и алгоритмов обработки текста (например, стандартные реализации алгоритмов стемминга и лемматизации для английского языка). Данное требование обусловлено тем, что в случае отсутствия соответсвующих средств необходимо будет их реализовать самостоятельно. Рекомендуется использовать язык программирования `python` либо `R`.

В процессе выполнения лабораторных работ требуется точно выполнять приведенные инструкции, особенно - указания по организации внутренней структуры директории проекта. Исходный код, сформированный в результате выполнения лабораторной работы, необходимо оформлять в соответствии с правилами, общепринятыми для той или иной платформы (для языка программирования `python` см. [пример оформления](/projects/emoji-labeller) проекта). Рекомендуется выделять функциональность, относящуюся к каждой отдельной лабораторной работе в отдельный программный модуль так, чтобы каждому модулю соответствовала отдельная директория в корневом каталоге проекта. Также рекомендуется привести краткую инструкцию по запуску исходного кода, являющегося результатом выполнения той или иной лабораторной работы, в файле `README.md`, расположенном в корневой директории проекта.

Помимо всего прочего, с целью упрощения сдачи лабораторных работ рекомендуется для каждой работы реализовать набор модульных тестов, демонстрирующих корректность выполнения задания. Инструкцию по запуску модульных тестов в таком случае также следует добавить в файл `README.md`, расположенный в корневой директории проекта.

## Порядок загрузки результатов выполнения лабораторных работ

В соответстии с [основным файлом README.md](/README.md) результаты выполнения лабораторных работ оформляются в виде проектов и размещаются в отдельных директориях внутри каталога [projects](/projects). Для этого необходимо выполнить следующие действия:

1. Сделать *fork* данного репозитория в свой аккаунт на `github`;
1. Придумать название проекта, **которое должно соответствовать конвенции kebab-case**, создать соответствующую директорию в каталоге [projects](/projects) и создать запись в таблице соответствия названий проектов и имен студентов в [основном файле README.md](/README.md);
1. Закоммитить изменения и сформировать `pull-request` в [основной репозиторий](https://github.com/MANASLU8/nlp-23-autumn);
1. После того, как первый `pull-request` принят - начинать выполнение лабораторных работ. Во время выполнения лабораторных работ требуется поддерживать определенную [структуру файлов](/projects/README.md) проекта, пример корректного оформления проекта можно посмотреть [здесь](/projects/emoji-labeller).
1. После защиты **каждой** лабораторной работы необходимо формировать новый `pull-request` в [основной репозиторий](https://github.com/MANASLU8/nlp-23-autumn) с обновленной версией проекта.
