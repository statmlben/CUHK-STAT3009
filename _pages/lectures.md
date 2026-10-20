---
layout: schedule
permalink: /lectures/
title: Schedule
description: Fall 2026. Additional lectures will be announced.
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

<tr class="{{ event_type }}{% if lecture.quiz %} schedule-quiz{% endif %}">
    <th scope="row">
        <span class="schedule-date-day">{{ lecture.date | date: "%a" }}</span>
        <span class="schedule-date-value">{{ lecture.date | date: "%m/%d" }}</span>
    </th>
    {% if lecture.recitation and lecture.recitation != empty %}
    {% assign recitation_count = recitation_count | plus: 1 %}
    {%endif%}
    {% if lecture.title contains 'No class' or lecture.title contains 'cancelled' or lecture.title contains 'Buffer' %}
        {% assign skip_classes = skip_classes | plus: 1 %}
        <td colspan="4" align="center">{{ lecture.title }}<p align="right">{{ lecture.logistics }}</p></td>
    {% elsif lecture.quiz and lecture.quiz != empty %}
        {% assign skip_classes = skip_classes | plus: 1 %}
        <td colspan="4" align="center">{{ lecture.quiz }}<p align="right">{{ lecture.logistics }}</p></td>
    {% else %}
    <td class="schedule-lecture">
        {% if lecture.title %}
            <span class="schedule-lecture-number">Lecture #{{ forloop.index | minus: current_module | minus: skip_classes | minus: recitation_count }}{% if lecture.lecturer %} · {{ lecture.lecturer }}{% endif %}</span>
            <div class="schedule-lecture-title">{{ lecture.title }}</div>
        {% endif %}
        {% if lecture.recitation %}
            <span class="schedule-lecture-number">Recitation #{{ recitation_count }}</span>
            <div class="schedule-lecture-title">{{ lecture.recitation }}</div>
        {% endif %}
        {% if lecture.pynb %}
            <a class="schedule-notebook" href="{{ lecture.pynb }}" target="_blank">Open notebook</a>
        {% endif %}
        {% if lecture.slides or lecture.pdf or lecture.slides2 or lecture.annotated or lecture.video or lecture.notes %}
        <div class="schedule-materials">
            {% if lecture.slides %}
              <a href="{{ lecture.slides }}" target="_blank">Slides</a>
            {% endif %}
            {% if lecture.pdf %}
              <a href="{{ lecture.pdf }}" target="_blank">PDF</a>
            {% endif %}
            {% if lecture.slides2 %}
              <a href="{{ lecture.slides2 }}" target="_blank">Slides 2</a>
            {% endif %}
            {% if lecture.annotated %}
              <a href="{{ lecture.annotated }}" target="_blank">Annotated</a>
            {% endif %}
            {% if lecture.video %}
              <a href="{{ lecture.video }}" target="_blank">Video</a>
            {% endif %}
            {% if lecture.notes %}
              <a href="{{ lecture.notes }}" target="_blank">Notes</a>
            {% endif %}
        </div>
        {% endif %}
    </td>
    <td class="schedule-readings">
        {% if lecture.warmup %}
            <a class="schedule-warmup" href="{{ lecture.warmup.href }}" target="_blank">
                <span class="schedule-warmup__label">Warm-up</span>
                <span class="schedule-warmup__title">{{ lecture.warmup.title }}</span>
                {% if lecture.warmup.meta %}<span class="schedule-warmup__meta">{{ lecture.warmup.meta }}</span>{% endif %}
            </a>
        {% endif %}
        {% if lecture.readings %}
        <ul>
        {% for reading in lecture.readings %}
            <li>{{ reading }}</li>
        {% endfor %}
        </ul>
        {% endif %}
    </td>
    <td class="schedule-logistics">
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
