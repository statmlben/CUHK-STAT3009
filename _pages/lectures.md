---
layout: schedule
permalink: /lectures/
title: Schedule
---

{% assign current_module = 0 %}
{% assign skip_classes = 0 %}
{% assign prev_date = 0 %}
{% assign recitation_count = 0 %}

{% for item in site.data.lectures %}
{% if item.date %}
{% assign lecture = item %}
{% assign event_type = "upcoming" %}
{% assign today_date = "now" | date: "%s" | divided_by: 86400 %}
{% assign lecture_date = lecture.date | date: "%s" | divided_by: 86400 %}
{% if today_date > lecture_date %}
    {% assign event_type = "past" %}
{% elsif today_date <= lecture_date and today_date > prev_date %}
    {% assign event_type = "warning" %}
{% endif %}
{% assign prev_date = lecture_date %}

<tr class="{{ event_type }}">
    <th scope="row">{{ lecture.date }}</th>
    {% if lecture.recitation != blank %} 
    {% assign recitation_count = recitation_count | plus: 1 %}
    {%endif%}
    {% if lecture.title contains 'No class' or lecture.title contains 'cancelled' or lecture.title contains 'Buffer' %}
        {% assign skip_classes = skip_classes | plus: 1 %}
        <td colspan="4" align="center">{{ lecture.title }}<p align="right">{{ lecture.logistics }}</p></td>
    {% elsif lecture.quiz != blank %}
        {% assign skip_classes = skip_classes | plus: 1 %}
        <td colspan="4" align="center">{{ lecture.quiz }}<p align="right">{{ lecture.logistics }}</p></td>
    {% else %}
    <td>
        {% if lecture.title %}
            Lecture #{{ forloop.index | minus: current_module | minus: skip_classes | minus: recitation_count}}
            {% if lecture.lecturer %}({{ lecture.lecturer }}){% endif %}:
        {% endif %}
        {% if lecture.title %}
            <br />{{ lecture.title }}<br />
        {% endif %}
        {% if lecture.recitation %}
            Recitation #{{ recitation_count }}:
        {% endif %}
        {% if lecture.recitation %}
            <br />{{ lecture.recitation }}<br />
        {% endif %}
        [
            {% if lecture.slides %}
              <a href="{{ lecture.slides }}" target="_blank">slides</a>
            {% endif %}
            {% if lecture.slides2 %}
              | <a href="{{ lecture.slides2 }}" target="_blank">slides 2</a>
            {% endif %}
            {% if lecture.annotated %}
              (<a href="{{ lecture.annotated }}" target="_blank">annotated</a>)
            {% endif %}
            {% if lecture.video %}
            | <a href="{{ lecture.video }}" target="_blank">video</a>
            {% else %}
            <!-- | video -->
            {% endif %}
            {% if lecture.notes %}
              <a href="{{ lecture.notes }}" target="_blank">notes</a>
            {% endif %}
            {% if lecture.pynb %}
              | <a href="{{ lecture.pynb }}" target="_blank">pynb</a>
            {% endif %}
        ]
    </td>
    <td>
        {% if lecture.readings %}
        <ul>
        {% for reading in lecture.readings %}
            <li>{{ reading }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </td>
    <td>
        <p>{{ lecture.logistics }}</p>
    </td>
    {% endif %}
</tr>
{% else %}
{% assign current_module = current_module | plus: 1 %}
{% assign module = item %}
<tr class="info">
    <td colspan="5" align="center"><strong>{{ module.title }}</strong></td>
</tr>
{% endif %}
{% endfor %}
