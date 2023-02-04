
# Publications

```{jinja} publications_ctx

<div class="disable-hyphens">
{% for p in publications %}
<b>{{p.title}}</b><br/>
{{p.authors}}<br/>
<em>{{p.conference}}</em><br/>
{% if p.paper %}<a href="{{p.paper}}">paper</a>{% endif %}
{% if p.code %}<a href="{{p.code}}">code</a>{% endif %}
{% if not loop.last %}<hr/>{% endif %}
{% endfor %}
</div>
```
