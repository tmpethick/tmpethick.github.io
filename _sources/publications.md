
# Publications

```{jinja} publications_ctx

<div class="disable-hyphens">
{% for p in publications %}
<a href="{{p.paper}}"><b>{{p.title}}</b></a><br/>
{{p.authors}}<br/>
<em>{{p.conference}}</em><br/>
{% if p.paper %}<a href="{{p.paper}}">paper</a>{% endif %}
{% if p.code %}<a href="{{p.code}}">code</a>{% endif %}
{% if not loop.last %}<hr/>{% endif %}
{% endfor %}
</div>
```
